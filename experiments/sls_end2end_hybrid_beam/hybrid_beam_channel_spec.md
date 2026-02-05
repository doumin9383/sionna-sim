# 実装仕様書: Sionna Hybrid Beamforming Extensions

## 概要

Sionna (v0.14+) を拡張し、サブパネル構成のハイブリッドビームフォーミング（HBF）に対応したチャネルモデルを実装する。
大規模MIMOにおけるVRAM枯渇（OOM）を防ぐため、全サブキャリアを一括で計算せず、チャンク分割して処理する仕組みを導入する。

## 前提条件 / 依存関係

* **Base Framework:** TensorFlow, Sionna v0.14+
* **Dependencies:**
* `SubPanelPlanarArray` (既存/別途実装): `get_analog_mapping()` や `port_positions` を提供するカスタムアレイクラス。
* `sionna.channel.OFDMChannel`: 継承元クラス。



---

## Class 1: `ChunkedOFDMChannel`

### 責務

* `sionna.channel.OFDMChannel` を継承し、周波数応答 () の生成プロセスを「サブキャリア単位のチャンク処理」に変更可能にする。
* Sionna標準の「パス生成 ()」ロジックはそのまま流用し、メモリを消費する「時間領域→周波数領域変換」の部分のみをスライスアクセス可能にする。

### メソッド仕様

#### `get_h_freq_chunk(self, start_idx, num_chunk)`

指定されたサブキャリア範囲の周波数応答のみを計算して返す。

* **Args:**
* `start_idx` (int): 開始サブキャリアインデックス。
* `num_chunk` (int): 取得するサブキャリア数。


* **Process:**
1. 親クラスの `get_channel_coefficients()` を呼び出し、パス利得 `h` と遅延 `tau` を取得する。
2. `self._frequencies` から `start_idx` 〜 `start_idx + num_chunk` の範囲をスライスする。
3. スライスした周波数を用いて位相回転  を計算する。
4. パス次元を縮約 (`reduce_sum`) し、 `[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_chunk]` の形状で返す。


* **Returns:**
* `tf.Tensor`: 指定範囲のチャネル行列 (Complex64/128)。



---

## Class 2: `HybridOFDMChannel`

### 責務

* `tf.keras.layers.Layer` を継承（またはラッパーとして実装）。
* アナログBF（サブパネル構造）の適用と、デジタル部から見た「実効チャネル ( or  )」の提供。
* **VRAM保護:** 内部で `ChunkedOFDMChannel` を保持し、`get_h_freq_chunk` をループで呼び出しながら、即座にアナログBF行列を掛けて次元を削減する。

### プロパティ / 属性

* `physical_channel`: `ChunkedOFDMChannel` のインスタンス。
* `tx_array`, `rx_array`: `SubPanelPlanarArray` のインスタンス。
* `_w_rf` (Variable): 送信アナログ重み `[num_tx_elements, num_tx_ports]`。
* `_a_rf` (Variable): 受信アナログ重み `[num_rx_elements, num_rx_ports]`。

### メソッド仕様

#### `__init__(self, physical_channel, tx_array, rx_array, ...)`

* 各アレイから `get_analog_mapping()` (マスク) を取得し、接続されていない素子の重みが操作されないよう初期化時のバリデーションやマスク保持を行う。

#### `set_analog_weights(self, tx_weights=None, rx_weights=None)`

* 外部（強化学習エージェントやスイープアルゴリズム）からアナログ重みを更新する。
* 入力された重みにハードウェア接続マスク（どの素子がどのポートか）を適用してから内部変数を更新すること。

#### `get_port_channel(self, chunk_size=64)`

デジタルPrecoder/Detectorが見るための「ポート単位の実効チャネル」を生成する。

* **Process:**
1. `physical_channel.num_subcarriers` を取得。
2. Pythonループ (`range(0, total_sc, chunk_size)`) で以下を実行:
a.  `physical_channel.get_h_freq_chunk(...)` を呼び出し、素子レベルのチャネル断片 `h_elem_chunk` を取得。
b.  `tf.einsum` を用いてアナログBFを適用し、次元を落とす。
* 計算式イメージ:
* Shapes: `[rx_elem, rx_port]^H` @ `[..., rx_elem, tx_elem, ...]` @ `[tx_elem, tx_port]`
c.  結果リストに追加。
3. `tf.concat` で周波数軸を結合して返す。


* **Returns:**
* `tf.Tensor`: `[batch, num_rx_ports, num_tx_ports, num_subcarriers]` (※Sionnaの標準出力形式に準拠)



#### `call(self, x)` (Option)

* 信号伝送シミュレーション用。
* 入力 `x` (デジタル信号) に対して  を適用 → `physical_channel` (素子レベル) →  で受信合成を行う。
* **Note:** ここでも全サブキャリア展開を避けるため、必要であれば内部で `physical_channel.apply_channel` の挙動を参考にしつつ、最適化されたパスを使用する。

---

## 実装上の注意点 (Tips for Coding Agent)

1. **Tensor Shapes:** Sionnaのチャネル係数は `[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]` という深い次元を持つため、`get_h_freq_chunk` 内での `reduce_sum` (パス合成) の軸指定を間違えないこと。
2. **Complex Conjugate:** 受信側のアナログ合成 () は、信号処理的には「重みの複素共役」を掛けて足す処理である。`tf.einsum` を書く際は `adjoint` の扱い（または入力前に `conj` するか）を明確にすること。
3. **Performance:** `get_port_channel` 内のループは `tf.function` 内で展開されると巨大なグラフになる可能性がある。`tf.map_fn` を検討するか、あるいはPythonのforループでもSionnaのEager Execution環境下では許容範囲か検討すること（推奨はPythonループでメモリ解放を優先）。