# 2026-11-10 视频侧小规模实验

两轮低成本验证（17M 缩小版模型，AutoDL 2×MTT S4000 / MUSA），目标是先判断 pp_zh §5.5
"从单帧到视频流"的两条结构性推断值不值得上全尺寸实验，**只做机制验证，不追求绝对重建质量**。

| 文档 | 内容 | 结论 |
|---|---|---|
| [REPORT_video_loop_adaptive_budget.md](REPORT_video_loop_adaptive_budget.md) | ①相邻帧连续假设：跨帧复用解码器 carry + 每帧按重建平台截断预算 | 自适应预算成立（−38% token / −0.1 dB）；carry 复用失败（同 token −3 dB，每 8 帧 GOP 漂移 ~2 dB） |
| [REPORT_intermediate_state_base.md](REPORT_intermediate_state_base.md) | ②既视感/长程假设：新增跨帧循环"中间态"生成解码器初始基底（编码器/解码器不变，整段 BPTT） | 接口有信息值（4 token/帧时上界 +4.7 dB、拷贝匹配历史帧 +3.3~3.7 dB），但当前实现部署口径下拿不到 Pareto 收益；瓶颈=注入点冲突 + 低预算历史质量反馈环 + 检索 cue 判别性不足 |
| [SURVEY_video_visual_token_prior_work.md](SURVEY_video_visual_token_prior_work.md) | 先验工作撞车检查（逐条核到原文链接） | 雷同度中高；最近 ElasticTok / AdapTok / ALIT / AdaCodec / PVC；中间态"生成解码器初始基底"的接口未找到先例 |

代码与数据产物保留在实验机本地（`/home/linaro/dsh/video_recon_probe/`）与远程
`/root/autodl-tmp/video_recon/`，不在本仓库内。
