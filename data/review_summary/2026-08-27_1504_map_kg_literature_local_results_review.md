# 2026-08-27 15:04 HKT MAP-KG 文献与 PTV3 本地结果联合审阅

## 审阅范围

- 阅读 `data/paper/2026.02.25.708091v1.full.pdf`（MAP: A Knowledge-driven Framework for Predicting Single-cell Responses for Unprofiled Drugs）。
- 检查 PTV3 当前图特征、图缓存、训练入口，以及 Exp01–09、Exp31–35 的结果和既有审阅记录。
- 检索 TxPert、PertKGE、scGSDR、RPath、SynergyGraph/HIG-Syn、MeTDDI、TransDRP 等相关工作。
- 本次仅做代码/结果审阅和方案设计，没有修改训练代码，也没有执行 smoke test。

## 论文要点与本地实现的差异

MAP-KG 汇总 14 个资源，去重后约 18.7 万药物节点、2.29 万基因节点和约 69.4 万条药物-基因/基因-基因关系；药物同时使用 SMILES 与文本，基因同时使用蛋白序列与文本，并用关系文本做关系条件化表示。该图谱是面向转录组单细胞响应的静态、全局机制先验。

当前 PTV3 的图特征是以药物为中心的压缩表示：PDI 蛋白投影、PPI 邻居传播、药物上下文以及结构随机投影。主缓存为约 `[6131, 774]`，另有多跳缓存；尚无 pathway/cell/assay 节点、激活/抑制符号、证据来源和时间/剂量上下文。`utils/05_build_graph_matrices_from_global_meta.py` 中名为 `ddi_matrix` 的矩阵由半径 2 Morgan 指纹的 Bulk Tanimoto 相似度构造，因此应被解释或重命名为 `drug_similarity_matrix`；它不能替代机制性 DDI、靶点重叠、CYP/转运体等关系。

## 本地结果给出的机会和风险

- 主单药任务中图特征确实有效：Exp01 图开关相对 no-graph 的 AUPRC 提升约 `+0.0733`，AUROC 提升约 `+0.0538`；Exp01/03 fold0 的 graph-zero AUPRC 分别下降约 `0.1051/0.0757`。
- 单药泛化明显好于双药：Exp06 未见药物组合 AUPRC `0.6398`，而 Exp01/02/03 单药 AUPRC 约 `0.6714/0.7914/0.7661`；Exp08 外部双药的 AUPRC 更低。这说明“组合机制交互”是比继续堆单药节点数更有价值的增量方向。
- 现有图谱缺少上下文和不确定性：global metadata 中约 1,137/6,130 个药物没有 target list，PDI 缓存中仍有 294 个药物未映射；静态正权重会把缺失边、弱证据边和真实无相互作用混在一起。
- 蛋白方向预测存在显著正向偏置：既有审阅显示 Exp01–06 在 `|delta|>0.1` 时下调召回率仅 `3.59–22.59%`，远低于上调；Exp09/31–34 的输出几乎单向为正。控制值缺失被置零且 MSE 有效掩码不对称，是在引入 pathway/delta 一致性损失前必须修复的问题。

## 建议包装的主线

建议将工作包装为 **PTV-CMK（Contextual Mechanism Knowledge Graph）**：面向蛋白组响应和药物组合的“上下文条件化、带符号、带证据、多模态机制图谱”。创新点不是与 MAP 比数据库大小，而是把静态药物-基因先验变成由细胞基线蛋白组（并可迁移到 RNA-seq）调制的个体化机制子图，并显式建模方向、证据和组合交互。

### P0：先修正审计与可解释性契约

1. 将 `ddi_matrix` 拆成 `drug_similarity` 与真正的 mechanistic DDI 通道；在配置、缓存元数据和报告中明确命名。
2. PDI/PPI 边保留 `rel_type`（binding/activation/inhibition/PPI 等）、`sign`、原始分数、来源、独立来源数、证据等级和时间戳；不要把所有边压成单一正权重。
3. 修复 `isfinite(control) & isfinite(target)` 的联合 MSE 掩码，单独提供缺失性 head；增加 copy-control、上/下调召回、符号平衡误差、Brier/ECE 等指标。KG 增益应在方向审计通过后再宣称。

### P1：低成本、可证伪的第一轮实验

在现有模型上新增 typed/provenance PDI/PPI 通道和 pathway pooling，不改变主干：

- 节点：Drug–Protein/Gene–Pathway–Cell/Assay。
- 边：`binds`、`inhibits`、`activates`、`PPI`、`in_pathway`、`expressed_in`、`response_to`；训练折之外的 response 边必须删除。
- 先做药物→靶蛋白→PPI→pathway 的多跳池化，比较当前图、+typed/provenance、+pathway 三个版本。
- 在 Exp01/03（未见药物/细胞）、Exp06（未见组合）上做严格消融；报告机制覆盖度分层结果和随机重连/删边 sanity check。

### P2：论文主贡献——上下文条件化机制子图

对每个样本/细胞基线蛋白组 (x_{p,c}) 重加权边：

`w(d→p | c) = w_e × g(log1p(x_{p,c})) × confidence_e`，并把 abundance missingness 作为显式特征而不是零值。用 context-gated message passing 或 personalized subgraph 得到药物机制表示；RNA-seq Exp31 使用 UniProt↔HGNC/Ensembl 桥接和模态专属 encoder，再对齐同一 Drug/Pathway 机制空间。这样既吸收 MAP/TxPert 的转录组知识，又保留 PTV3 的蛋白组优势。

增加 pathway-level auxiliary loss：由预测的蛋白 delta 计算 pathway activity/direction，与 KG 传播出的 pathway effect 做一致性约束；输出 pathway attention、关键路径和证据等级。该损失必须在方向掩码修复后启用。

### P3：双药/协同的超图交互头

把组合表示分解为 `delta_AB = delta_A + delta_B + interaction_AB`，只让交互头学习残差；超边可为 `(drug_A, drug_B, cell, pathway/module)`。特征包括靶点邻域 Jaccard、pathway 覆盖/互补、最短路径、符号冲突或一致、剂量/顺序。用 hypergraph 或 pair cross-attention，并在 Exp06/外部 Exp08 上验证；Exp33 只能作为无标签排序，不能当作 synergy accuracy。

## 建议的验证矩阵

| 版本 | Exp01/03 | Exp06 | Exp31 | Exp32/33/35 |
|---|---|---|---|---|
| 当前图（基线） | 复核 | 复核 | BRCA→非 BRCA | 仅排序 |
| typed + provenance | 主消融 | 主消融 | 迁移性 | 机制置信度 |
| + pathway/context gating | 主结果 | 次结果 | RNA/蛋白对齐 | pathway/不确定性排序 |
| + hypergraph residual | 不适用 | 主结果 | 不适用 | 组合候选 |

统一报告 AUROC/AUPRC、delta PCC/MSE、copy-control、上/下调召回、pathway GSEA/overlap、校准误差和按机制覆盖度分层的性能。对 Exp32/33/35 输出 `response probability × mechanism confidence`，并保留“无标签、未校准、OOD”的声明。

## 文献依据

- TxPert 表明 STRING、GO、PxMap、TxMap 等互补图谱联合后性能继续提升，并指出噪声/不完整边需要注意力、长程图传播和 provenance-aware 的多图策略。
- PertKGE 说明用扰动转录组反推 compound–protein 关系，可改善冷启动靶点推断和虚拟筛选；这适合作为 PTV-CMK 的“实验响应反哺边权”支路，但必须按训练折隔离以防泄漏。
- scGSDR 与 RPath 支持把 pathway/因果路径作为响应预测和解释层，而不是只做节点 embedding。
- SynergyGraph/HIG-Syn 支持用药物-药物-细胞-机制模块超边表达互补性；MeTDDI 支持用 motif/local-global attention 解释组合关系。
- TransDRP 的临床类别对齐可作为 Exp31 的后续患者域适配；PREDIKTOR 等最新预印本提示“患者特异网络 + 扰动视图对齐”是可延伸方向，但不应作为当前主要证据。

## 结论

最值得先做的不是复制 MAP 的超大数据库，而是把 PTV3 现有已被消融证明有效的图信号升级为：**有方向、有证据、由细胞基线调制、能落到 pathway，并对双药交互建模**。最小可行路线是 P0 审计修复 → P1 typed/provenance + pathway 消融 → P2 context-gated PTV-CMK → P3 hypergraph synergy；每一步都可在现有 Exp01/03/06 上证伪，且能解释 Exp31–35 的跨模态/OOD 结果。
