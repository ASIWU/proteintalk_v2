
# Step4：训练和推理代码需求说明

现在数据处理和数据划分都已经结束。请基于新的数据格式，编写 training 和 inference 代码。

---

## 1. 训练代码：train.py

请参考以下文件及其引用的库，写出一份新的 `train.py`：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train.py`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/model`
- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/dataset`

原本的训练脚本也可以作为参考：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/scripts/old/1225`

### 1.1 代码复用要求

代码尽可能复用原有实现，但需要适配新的数据格式。

### 1.2 新数据格式相关改动

需要注意以下几点：

1. 在数据处理中，相关的 `batch_cov_list` 已经完成 token 处理。
   - 因此这里应该不再需要 `embedding_methods`。

2. `batch_cov_list` 仍然保留现有形式。
   - 不需要引入 `pert_dose`。

3. `inverse_machine_id` 默认存在。
   - 删除这个参数。

---

## 2. 与原始训练代码不同的地方

### 2.1 只保留指定模型

原始代码中探索了许多模型，但这些模型现在是冗余的。

2026-05-07 更新：根据后续确认，新的代码中只需要保留以下两个 active model：

- `attention_v10_hetero_cls_ee`
- `baseline_emb_v3`

其中 `attention_v10_hetero_cls_ee` 是 consolidated PDI hetero graph model：

- 默认使用 target protein token；仅在显式调试时才使用 `--no-use-target`
  关闭；
- 是否使用 gene/protein gate 由参数控制；
- 不再拆分 `no_target` / `gate` 等多个 graph model name。

### 2.2 graph 默认支持 pdi

在 graph 相关逻辑中，默认支持 `pdi`。

请删除其他所有情况。

### 2.3 重要更新：所有模型默认支持 double_drug

请修改所有模型，使其默认支持 `double_drug`。

参考以下文件及其引用库：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train_dd.py`

注意：

1. 模型架构主要仍以 `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train.py` 为主。
2. `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/train_dd.py` 只作为 double drug 相关逻辑的参考。

具体要求：

1. 当前默认逻辑需要从单药改为双药。
2. 在 `pert_id2` 这一列的基础上，模型需要额外加入 `pert_id1`。
3. 这样 `single_drug` 和 `double_drug` 可以共用同一份代码。
4. 对 single-drug rows，数据处理必须写成 `pert_id2 == pert_id1`，
   不能把第二槽位填成 `"no"`。

### 2.4 group_size 处理

新的 training-ready dataset/model contract 永久忽略旧代码中的
`group_size` 维度。每个 dataset item 只包含一个 control row 和一个
perturb row；DataLoader batch 直接形成 `(batch_size, n_genes)` 和
`(batch_size, 2)` 的 no-group 输入。

补充说明：

- 如果这里处理数据不方便，需要修改 `data_process_1` 或者 `data_process_2`，请先不要修改。
- 请把需要修改的地方写入文档中告诉我，我后续会 check。

### 2.4 check 模型逻辑优化

原本 check 模型的逻辑应该可以写得更简洁、更高效。

你可以先实际执行或检查现有逻辑，然后把优化建议写入文档中告诉我。

---

## 3. 推理代码：infer.py

请参考以下文件及其引用的库，写出一份新的 `infer.py`：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/infer.py`

推理代码主要针对以下数据：

- `extra_singledrug`
- `extra_doubledrug`

原本的推理脚本可以作为参考：

- `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/scripts/old/infer/infer_all_0110_1.sh`
