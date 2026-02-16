# 技術負債・将来のタスク (Technical Debt & Future Tasks)

## 1. チャネルパラメータの Dataclass 移行
- **現状**: `MeshToSLSAdapter` および `HybridChannelInterface` は Tensor の辞書 (`dict`) を使用してデータをやり取りしています。
- **課題**: 辞書形式は柔軟ですが、型安全性が低く、キー名の不一致などのエラーが発生しやすいです。
- **提案**: `wsim.common.channel.interface` 等で共通の `ChannelParameters` Dataclass を定義し、SLS/LLS 全体で統一的なインターフェースに移行することを推奨します。
- **メモ**: ユーザーからの「dataclassのほうが柔軟？」という指摘に基づき、将来的な移行を検討すること。

## 2. LSP計算の一般化
- **現状**: `ExternalPaths` 内で `path_gain` からパスロスを逆算するロジックを実装しましたが、これは `Ptx = 1W` の前提に基づいています。
- **課題**: 将来的に異なる前提のレイトレデータが来た場合に再調整が必要です。
- **提案**: キャリア周波数や送信電力の前提をメタデータとしてより厳密に扱う仕組みを検討してください。
