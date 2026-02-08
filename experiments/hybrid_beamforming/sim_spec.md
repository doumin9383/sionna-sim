# **Simulation Specification: MIMO Precoding & PAPR/MPR Evaluation**

## **1\. 目的と概要**

本シミュレーションの主目的は、CP-OFDM および DFTS-OFDM (SC-FDMA) 波形を用いた MIMO 伝送において、**プリコーディングの適用粒度（Granularity）** が **PAPR (Peak-to-Average Power Ratio)** および **MPR (Maximum Power Reduction)** に与える影響を定量評価することである。
最終的に、MPR による送信電力バックオフ（Back-off）を考慮した上で、最もスループット（SE: Spectral Efficiency）が高くなる伝送設定（波形、プリコーディング粒度、ランク数）を明らかにする。

## **2\. 評価対象パラメータ**

### **2.1. 波形 (Waveform)**

* **CP-OFDM**: 標準的なOFDM。PAPRは高いが、周波数領域等化が容易。
* **DFTS-OFDM**: Transform Precoding (DFT) を適用した波形。低PAPR特性を持つが、MIMOプリコーディング適用時の特性劣化を評価対象とする。

### **2.2. プリコーディング粒度 (Precoding Granularity)**

チャネル情報（CSI）に基づいて計算するデジタルプリコーディング行列 $W\_{BB}$ の更新頻度。

1. **Wideband (WB)**: 全帯域で同一のプリコーディング行列を適用。
2. **Subband / RBG (Resource Block Group)**: 指定したサブバンド単位（例: 2RB, 4RB）で更新。
3. **Narrowband / RE**: サブキャリア単位（RE）で更新（理想SVD）。
   * *仮説*: 粒度が細かいほど理論的なビームフォーミング利得は上がるが、周波数領域でのスペクトル操作が複雑になり、DFTS-OFDM等の低PAPR特性を損なう可能性がある。

### **2.3. ハイブリッドBF構成**

* **Analog Beam**: 固定または Codebook ベースで選択。
* **RF Chain / Digital Ports**: 物理アンテナ数よりも少ないRFチェーン数で動作。
  * *評価点*: 制限されたRFチェーン下で、どの程度のデジタルプリコーディング粒度が適切か。

## **3\. シミュレーションフロー**

本シミュレーションは、**Phase 1 (LLS: 事前計算)** と **Phase 2 (SLS: システム評価)** の2段階で構成する。

### **Phase 1: LLS (Link Level Simulation) \- PAPR/MPR テーブル作成**

物理層パラメータごとに波形を生成し、PAPR特性を取得してMPRテーブル化する（事前計算）。

1. **パラメータスイープ**:
   以下の組み合わせについてループを実行する。
   * 波形: CP-OFDM, DFTS-OFDM
   * 変調方式: QPSK, 16QAM, 64QAM, 256QAM
   * ランク数: 1, 2, ..., $N\_{stream}$
   * プリコーディング粒度: WB, Subband, RE
   * 割り当て帯域幅 ($M\_{RB}$): 最小〜最大
2. **信号生成**:
   * ランダムビット生成 $\\rightarrow$ 変調 $\\rightarrow$ (DFT) $\\rightarrow$ プリコーディング $\\rightarrow$ IFFT $\\rightarrow$ 時間波形生成。
   * ※ プリコーディング行列はランダム、または代表的なチャネルモデル（TDL等）から生成したものを使用。
3. **PAPR 測定 & MPR 決定**:
   * 生成した時間波形の CCDF を計算し、99.9% 点の PAPR を取得。
   * 基準値あるいは 3GPP テーブルに基づき、必要な **MPR (dB)** を決定。
4. **ルックアップテーブル (LUT) 出力**:
   * Key: {Waveform, Modulation, Rank, Granularity, Bandwidth}
   * Value: {MPR \[dB\]}

### **Phase 2: SLS (System Level Simulation) \- スループット評価**

Phase 1 で作成したテーブルを参照し、マルチセル環境での実効スループットを評価する。

#### **Step 1: チャネル生成 & 等価チャネル算出**

1. 空間チャネルモデル (TR38.901) から $H\_{phy}$ を生成。
2. アナログ重み $W\_{RF}, A\_{RF}$ を適用し、デジタルポート間の等価チャネル $H\_{eff}$ を取得。
   $$H\_{eff}\[k\] \= A\_{RF}^T H\_{phy}\[k\] W\_{RF}$$

#### **Step 2: リンクアダプテーション & パラメータ選択 (MPR考慮)**

各ユーザについて、以下の手順で最適な送信設定を探索する。

1. **基準 SINR 推定**:
   基準電力（$P\_0$ ベース）でパイロット信号を送信した場合の推定 SINR を計算。
2. **候補パラメータのループ**:
   候補となる {波形, ランク, 粒度, MCS} の組み合わせごとに以下を計算：
   * **MPR 参照**: Phase 1 のテーブルから、当該設定に対応する MPR を取得。
   * **送信電力決定 (**$P\_{tx}$**)**:
     $$P\_{CMAX} \= P\_{PowerClass} \- \\text{MPR}$$$$P\_{tx} \= \\min \\left\\{ P\_{CMAX}, \\ P\_0 \+ 10 \\log\_{10}(M\_{RB}) \+ \\alpha \\cdot PL \\right\\}$$
   * **実効 SINR 算出**: $P\_{tx}$ に基づいて SINR を補正。
     $$\\text{SINR}\_{eff} \= \\text{SINR}\_{ref} \+ (P\_{tx} \- P\_{ref}) \- \\text{PrecodingLoss}(\\text{Granularity})$$
     ※ プリコーディング粒度によるビームフォーミング利得/損失は、チャネル $H\_{eff}$ とプリコーダ $W\_{BB}$ の内積から正確に計算する。
3. **最適設定の決定**:
   実効 SINR と BLER モデルに基づき、最大スループットが得られる組み合わせを選択。

#### **Step 3: スループット算出**

選択された設定（実際に適用する $P\_{tx}$, MCS, Rank）を用いて、そのスロットでのスループットを記録する。

## **4\. 必要な実装コンポーネント**

| **コンポーネント** | **機能概要** | **新規/既存** |
| HybridSystemSimulator | 全体制御。PAPR評価ループの実装。 | **改修** |
| PrecoderGenerator | 粒度(WB/SB/RE)を指定して $W\_{BB}$ を生成するクラス。 | **新規** |
| WaveformEvaluator | DFTS/CP切り替え、IFFT後のPAPR計算機能。 | **新規** |
| PowerControl | $\\alpha, P\_0, PL$ と $P\_{CMAX}$ から $P\_{tx}$ を計算するクラス。 | **新規** |
| LinkAdaptationWithMPR | $P\_{tx}$ を考慮した電力でSINRを再計算し、MCSを選ぶ。 | **新規** |
| MPRModel | PAPR値からMPR(dB)を返す関数/テーブル。 | **新規** |

## **5\. 出力データ / グラフ**

### **LLS (Link Level Simulation)**

単一リンク環境で、物理パラメータの影響を評価する。

1. **PAPR の CCDF (Complementary Cumulative Distribution Function)**
   * **目的**: 波形（CP vs DFTS）およびプリコーディング粒度（WB vs SB vs RE）が PAPR 特性に与える影響を確率分布として可視化する。
   * **軸**: 横軸 PAPR (dB), 縦軸 Probability ($P(PAPR \> X)$)。
2. **Throughput vs Pathloss (SNR)**
   * **目的**: 特定のパスロス（＝受信SNR）において、各方式がどの程度のスループットを達成できるかを比較する。ここではMPRを適用した後の実効電力を用いる。
   * **軸**: 横軸 Pathloss (dB) または SNR (dB), 縦軸 Throughput (Mbps)。
3. **Optimal Granularity Map**
   * **目的**: SNR（またはPathloss）ごとに、最も高いスループットを達成できる最適なプリコーディング粒度を示す。
   * **軸**: 横軸 SNR/Pathloss, 縦軸 最適粒度（WB/SB/RE）。

### **SLS (System Level Simulation) \- Simplified**

マルチセル環境（1セル1ユーザのランダムドロップ想定）で、システム全体の統計的性能を評価する。

1. **Throughput CDF (Cumulative Distribution Function)**
   * **目的**: セル内の様々な場所にランダムに配置されたユーザに対して、リンク/ランクアダプテーション（Open-loop Power Control & MPR考慮）を行った結果得られるスループットの分布を示す。
   * **軸**: 横軸 Throughput (Mbps), 縦軸 CDF。
   * **洞察**: セル端（低SNRかつ高パワー送信が必要な領域）でのDFTS-OFDMの優位性や、セル中心でのCP-OFDM/MIMOの優位性を確認する。
2. **Adaptation Statistics (アダプテーション統計)**
   * **目的**: 選択されたMCS、ランク、プリコーディング粒度の分布を示す。
   * **内容**:
     * ランク分布（Rank 1 vs Rank 2 vs ...）のヒストグラム。
     * 選択されたMCSインデックスのヒストグラム。
     * （適応的に粒度を変える場合）選択されたプリコーディング粒度の割合。