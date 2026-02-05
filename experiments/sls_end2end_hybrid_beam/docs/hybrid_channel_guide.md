## アーキテクチャとコードの対応関係

本ライブラリの各クラスが、ハイブリッドビームフォーミングのどのコンポーネントに対応しているかを以下の図に示します。

```mermaid
graph TD
    subgraph "Digital Domain (Sionna Standard)"
        DS[ResourceGrid / StreamManagement]
        DP[Digital Precoding/Equalization]
    end

    subgraph "HybridOFDMChannel (Digital-to-Analog Bridge)"
        direction TB
        subgraph "Internal Processing"
            PC["Physical Channel<br/>(GenerateOFDMChannel)"]
            CH["Chunked Processing Loop<br/>(VRAM Optimization)"]
            BF{"Analog Weight Application<br/>(tf.einsum)"}
        end
        WRF["W_RF / A_RF Weights<br/>(set_analog_weights)"]
    end

    subgraph "Physical Domain (Antenna Theory)"
        PA[PanelArray / Antenna Elements]
        CH_MOD[TR38901 Channel Models<br/>UMa / UMi / RMa]
    end

    %% Connections
    DS -->|Port Indices| HybridOFDMChannel
    DP -->|Effective H_port| DS

    PC -->|CIR (a, tau)| CH
    CH -->|H_elem (Element-level)| BF
    WRF -.->|Apply| BF
    BF -->|H_port (Digital Ports)| DP

    CH_MOD -->|Geometry/Pathloss| PC
    PA -->|Array Geometry| PC

    style HybridOFDMChannel fill:#f9f,stroke:#333,stroke-width:2px
    style Digital Domain fill:#bbf,stroke:#333
    style Physical Domain fill:#bfb,stroke:#333
```

## 各コンポーネントの役割

### 1. HybridOFDMChannel (本クラス)
*   **役割**: デジタルポートと物理アンテナの「翻訳者」です。
*   **入力**: `batch_size`
*   **出力**: デジタルポート単位の周波数応答 $H_{port}$
*   **内部動作**: `tf.einsum` を用いて、物理チャネル $H_{elem}$ に対してアナログ重み $H_{port} = A_{RF}^H H_{elem} W_{RF}$ を適用します。

### 2. GenerateOFDMChannel (継承元)
*   **役割**: 物理的なマルチパス特性（CIR: Channel Impulse Response）を周波数ドメインの素子レベルチャネル $H_{elem}$ に変換します。
*   **最適化**: `HybridOFDMChannel` ではこれをラップし、チャンク分割生成を行うことで大規模MIMO実行時のメモリ消費を劇的に抑えています。

### 3. W_RF / A_RF (アナログ重み)
*   **役割**: RF位相シフタを表します。
*   **設定**: `set_analog_weights(w_rf, a_rf)` メソッドで設定。
*   **形状**: `[num_elements, num_ports]`。

## 主な特徴

-   **VRAM 最適化 (Chunked Processing)**: 全サブキャリアを一括でアンテナ素子展開すると VRAM が枯渇するため、内部でサブキャリアを分割して（デフォルト 72 サブキャリアずつ）ビームフォーミングを適用し、次元削減した後に結合します。
-   **Sionna 互換性**: `GenerateOFDMChannel` を継承しているため、既存の `apply_ofdm_channel` や `LSChannelEstimator` などにそのまま渡すことができます。

## 使い方

### インスタンス化

```python
from hybrid_channels import HybridOFDMChannel

# Generator の作成
hybrid_channel = HybridOFDMChannel(
    channel_model,      # UMa, UMi 等
    resource_grid,      # ResourceGrid
    tx_array,           # 送信側 PanelArray
    rx_array,           # 受信側 PanelArray
    num_tx_ports=4,     # デジタルポート数 (送信)
    num_rx_ports=1      # デジタルポート数 (受信)
)
```

### ビーム重みの設定

デフォルトでは Identity-like な重み（最初の数素子のみ使用）が設定されています。全素子を活用するには、重みを明示的に設定します。

```python
# W_RF: [num_ant, num_ports] の複号行列
hybrid_channel.set_analog_weights(w_rf, a_rf)
```

### チャネルの生成

標準の Block と同様に `__call__` で実行します。

```python
# 戻り値は [batch, num_rx, num_rx_ports, num_tx, num_tx_ports, num_ofdm, num_sc]
h_port = hybrid_channel(batch_size)
```

## 注意点

-   **ポート数とアンテナ数**: デジタル側のロジック（例えば `StreamManagement`）で設定するアンテナ数は、物理素子数ではなく、ここで指定した `num_ports` と一致させる必要があります。
-   **JITコンパイル**: 本 Block は内部にループ（チャンク処理）を含むため、初回実行時の TensorFlow XLA コンパイルに時間がかかる場合があります。
