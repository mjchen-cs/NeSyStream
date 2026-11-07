# 主观实验（ACR 5 分制）

## 文件
| 文件 | 作用 |
|---|---|
| `dashjs_sessions.json` | dash.js 真实播放器中的 120 个会话（4 种方法 × 30 条测试 trace）：每个分片的码率档位、启动时延、卡顿 |
| `make_stimuli.py` | 按固定种子分层抽取 6 条 trace × 4 种方法 = 24 段测试视频 + 2 段练习视频，并按会话日志渲染（卡顿 = 冻结画面 + 加载图标） |
| `stimuli/` | 渲染好的视频 `stim_XX.mp4` / `train_XX.mp4`（文件名不含方法），`stimuli.json` 是对照表（实验时不要打开），`playlist.json` 供评分程序使用 |
| `serve.py` + `app/index.html` | 评分程序（本地网页） |
| `analyze.py` | 数据分析 |
| `experimenter_log.csv` | 实验人员记录表（纸质或电子） |

## 实验当天
1. 用于评测的电脑：Windows + Chrome，显示器设为原生分辨率（1920×1080），关闭通知和自动更新，屏幕亮度固定。
2. 在 `code` 目录下运行：`python subjective/serve.py`，用 Chrome 打开 `http://localhost:8000`。
3. 受试者签署知情同意书后，实验人员在第一页填写**受试者编号**（S01、S02…，按到场顺序）和基本信息（年龄段、性别、观看频率、视力），点"开始"后交给受试者。
4. 程序自动完成：说明 → 2 段练习 → 12 段测试 → 休息（≥3 分钟）→ 12 段测试 → 结束。每段视频先完整载入内存再播放，不会因读盘产生额外卡顿；视频无进度条、不可拖动。测试顺序按编号随机，且相邻两段不来自同一条 trace。
5. 结束后在 `experimenter_log.csv` 记一行（编号、日期、开始/结束时间、是否完成、异常情况）。**不记录姓名**；姓名只出现在纸质知情同意书上，由负责人单独保管。

## 数据如何记录
- 每次打分立即追加到 `subjective/data/ratings.csv`（一行一次打分），程序中途关闭也不会丢失已打的分；每名受试者结束时另存 `subjective/data/Sxx.json`。
- `ratings.csv` 列：subject, age_group, gender, viewing_freq, vision, phase(train/test), trial, file, rating(1–5), rt_s（反应时）, played_s, complete, time。
- 中途退出：保留 `ratings.csv` 中已有数据；如受试者要求删除，删除该编号的所有行和 `Sxx.json`。
- 每天实验结束后把 `subjective/data/` 备份到加密的课题组电脑。

## 分析
```
python subjective/analyze.py --ratings subjective/data/ratings.csv --stimuli subjective/stimuli/stimuli.json
```
输出：按 ITU-T P.913 剔除异常受试者（与其他人平均分的相关系数 < 0.75），各方法 MOS 及 95% 置信区间（全部 / 3G与宽带 / 4G/5G），NeSyStream 与各方法的配对差异（Wilcoxon 检验 + bootstrap 置信区间），以及 P.1203 与 MOS 的相关性。结果写入 `subjective/data/results.json`。

## 重新渲染
```
python subjective/make_stimuli.py --sessions subjective/dashjs_sessions.json --src_dir deploy_dashjs/src --out subjective/stimuli
```
需要 dash.js 实验用的 13 档 60 秒源文件（`deploy_dashjs/src/level_XX.mp4`）和 ffmpeg。
