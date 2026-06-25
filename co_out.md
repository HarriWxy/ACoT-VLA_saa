# Train & infer

先定位 acotvla 和 pi 模型各自的实现入口，再沿着前向调用和策略封装看输入输出是在哪里约束的。

Searched for regex `acot_vla|pi0|class Pi0|class ACoT|forward\(|infer|sample_actions|Action|observation` (`**/src/openpi/**/*.py`), 200 results

Searched for regex `acotvla|pi0|ACoT|policy|serve|inference` (`**/README.md`), 100 results

Read memory [](file:///home/omnisky/.config/Code/User/workspaceStorage/f3b3bbf6136ff0068835e337cafa05d5/GitHub.copilot-chat/memory-tool/memories/repo/ACoT-VLA_saa.md)

已经定位到控制路径了：模型本体通过 model config 的 input spec 和前向函数定义张量形状，训练/推理侧再用 transforms 和 policy wrapper 把原始观测封装成模型输入。下一步我直接读这几处核心实现。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/model.py#1-1), lines 1 to 260

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/pi0.py)

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/acot_vla.py#1-1), lines 1 to 420

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/pi0_config.py)

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/data_loader.py#80-80), lines 80 to 170

模型接口本身已经清楚了：input spec 在 config 里，统一数据结构在 Observation 里，真正输出动作的是 sample_actions。现在补齐 ACoT_VLA 的双专家实现和 policy 入口，这样能把“定义”和“落地执行”串起来。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/model.py#260-260), lines 260 to 333

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/acot_vla.py#320-320), lines 320 to 760

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/policies/srb_policy.py)

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/transforms.py#1-1), lines 1 to 380

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/policies/policy.py)

还差 ACoT_VLA 的 loss 和采样收尾部分，我把这一段补全后就能直接给你完整链路。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/acot_vla.py#760-760), lines 760 to 901

Searched for regex `ACOTConfig|acot_icra_simulation_challenge_reasoning_to_action|coarse_actions|ModelType\.ACOT|sample_actions\(` (`**/src/openpi/**/*.py`), 32 results

Searched for regex `ModelTransformFactory|TokenizePrompt|PadStatesAndActions|Group\(|outputs=|Unnormalize\(|AbsoluteActions\(` (`**/src/openpi/**/*.py`), 74 results

输入侧还要补一层：训练/推理配置把原始数据映射到 Observation。输出侧则分成标准 PI 单路动作和 ACoT 的 coarse/fine 双路动作，我把这两层配置和调用关系也读一下。

Searched for regex `ACOTConfig\(|acot_icra_simulation_challenge_reasoning_to_action|coarse_action_horizon|adopt_explicit_action_reasoner|adopt_implicit_action_reasoner` (`**/src/openpi/**/*.py`), 32 results

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/config.py#100-100), lines 100 to 180

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/config.py#360-360), lines 360 to 410

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/policies/arx_policy.py#150-150), lines 150 to 207

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/policies/policy_config.py)

我顺手查一下 ACoT 配置有没有接进训练/推理配置注册里，这能判断你在代码里该从哪条入口看。

Searched for regex `acot_icra_simulation_challenge_reasoning_to_action|ACOT_VLA|ACOTConfig|acot` (`**/src/openpi/**/*.py`), 38 results

Searched for regex `acot_icra_simulation_challenge_reasoning_to_action|ACOT_VLA|acot` (`**/scripts/**/*.py`), 14 results

Searched for regex `acot_icra_simulation_challenge_reasoning_to_action|ACOT_VLA|acot` (`**/README.md`), 23 results

训练端对 ACoT 有一条单独分支，我再补这一个入口，你就能看到 coarse_actions 是怎么进损失函数的。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/scripts/train.py#180-180), lines 180 to 330

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/scripts/serve_policy.py#60-60), lines 60 to 110

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/scripts/openloop.py)

我补一组精确行号，方便你直接点进去看对应实现。

Searched for regex `class Observation|def from_dict|class BaseModelConfig|def inputs_spec|class BaseModel|def sample_actions|def preprocess_observation` (`**/src/openpi/models/model.py`), 7 results

Searched for regex `class Pi0Config|def inputs_spec|class Pi0|def embed_prefix|def embed_suffix|def compute_loss|def sample_actions` (`**/src/openpi/models/pi0.py`), 5 results

Searched for regex `class ACOTConfig|def inputs_spec|class ACOT_VLA|def embed_prefix|def embed_suffix|def compute_loss|def sample_actions` (`**/src/openpi/models/acot_vla.py`), 7 results

Searched for regex `class ModelTransformFactory|case _model.ModelType.PI0|case _model.ModelType.PI05|case _model.ModelType.PI0_FAST|class SRBDataConfig` (`**/src/openpi/training/config.py`), 5 results

Searched for regex `class Policy|def infer|class SRBInputs|class SRBOutputs|def create_trained_policy` (`**/src/openpi/{policies,training}/**/*.py`), 8 results

最后补一下 config 和 transforms 的精确位置，这两处就是“输入输出在哪定义”的直接答案。

Searched for regex `class Pi0Config|def inputs_spec` (`**/src/openpi/models/pi0_config.py`), 2 results

Searched for regex `class RepackTransform|class Normalize|class Unnormalize|class TokenizePrompt|class TokenizeFASTInputs|class PadStatesAndActions|class AbsoluteActions` (`**/src/openpi/transforms.py`), 7 results

Searched for regex `def acot_train_step|ACOT_VLA_PI05|ACOT_VLA_PI0` (`**/scripts/train.py`), 3 results

Searched for regex `DEFAULT_CHECKPOINT|acot_icra_simulation_challenge_reasoning_to_action` (`**/scripts/serve_policy.py`), 4 results

Searched for regex `coarse_actions|class ARXACOTInputs|class ARXACOTOutputs` (`**/src/openpi/policies/arx_policy.py`), 5 results

我再补一眼具体 config 的注册位置，避免你只看到模型类但找不到实际启动时用的是哪个配置对象。

Searched for regex `acot_icra_simulation_challenge_reasoning_to_action|acot_libero_action_cot_explicit_implicit_co_fusion|get_config\(` (`**/src/openpi/training/**/*.py`), 3 results

Searched for regex `ACOT_VLA_PI05|ACOT_VLA_PI0` (`**/src/openpi/models/model.py`), no results

Searched for regex `CoarseActions|Actions =` (`**/src/openpi/models/model.py`), 1 result

我顺手核对一下 CoarseActions 这个类型别名是不是在公共接口里也补齐了。

Searched for text `CoarseActions` (`**/src/openpi/**/*.py`), 1 result

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/config.py#1030-1030), lines 1030 to 1062

ACoT 的配置应该在大配置表里，我直接按文件精确搜一下。

Searched for text `acot_` (`**/src/openpi/training/config.py`), no results

Searched for regex `ACOTConfig\(` (`**/src/openpi/training/config.py`), no results

搜一下配置名在整个仓库里的落点，确认 ACoT 是通过哪份 TrainConfig 实例化的。

Ran terminal command:  rg -n "acot_icra_simulation_challenge_reasoning_to_action|acot_libero_action_cot_explicit_implicit_co_fusion|ACOTConfig\(" src scripts

本机没装 rg，我用工作区搜索工具直接限定到配置文件里查。

Searched for text `acot` (`ACoT-VLA_saa` · `src/openpi/training/config.py`), no results

Searched for text `ACOTConfig` (`ACoT-VLA_saa` · `src/openpi/training/config.py`), no results

我换成找文件名，避免被搜索过滤器挡住。

Searched for files matching `**/src/openpi/**/*acot*`, 1 match

我用系统自带 grep 再确认一遍配置名是不是只出现在脚本里。

Ran terminal command:  grep -RIn "acot_icra_simulation_challenge_reasoning_to_action\|acot_libero_action_cot_explicit_implicit_co_fusion\|ACOTConfig(" src scripts README.md

我再把 ACoT 返回值和通用 Policy 封装那两行定位出来，这样你能看出当前代码里接口是不是完全统一。

Searched for regex `return \{"actions": x_0_expert, "coarse_actions": explicit_action_reason\}|return \{"actions": x_0_expert\}` (`**/src/openpi/models/acot_vla.py`), 2 results

Searched for regex `"actions": self\._sample_actions` (`**/src/openpi/policies/policy.py`), 1 result

Searched for text `coarse_actions` (`**/src/openpi/training/data_loader.py`), no results

Searched for text `coarse_actions` (`**/examples/srb/README.md`), 2 results

我把 ModelType 这处也核一下，看看 ACoT 在公共枚举里是不是完整暴露出来了。

Searched for regex `class ModelType|PI0 =|PI0_FAST =|PI05 =` (`**/src/openpi/models/model.py`), 4 results

Searched for regex `return _model.ModelType.ACOT_VLA_PI0|return _model.ModelType.ACOT_VLA_PI05` (`**/src/openpi/models/acot_vla.py`), 2 results

如果你还关心 PI0-FAST，我补一下 token 输出转回动作的那一行。

Searched for text `class ExtractFASTActions` (`**/src/openpi/transforms.py`), 1 result

Searched for text `case _model.ModelType.PI0_FAST` (`**/src/openpi/training/config.py`), 1 result

如果你说的 “pi” 是这里的 Pi0 或 Pi05，那么这套仓库里“模型输入输出在哪定义”和“代码怎么跑起来”可以直接看三层：公共数据结构、模型配置、推理封装。

**定义层**
- 公共输入结构定义在 model.py、model.py 和 model.py。这里统一规定模型吃的是一个 Observation，核心字段是 “image / image_mask / state / tokenized_prompt / tokenized_prompt_mask”；标准动作输出别名是 model.py，形状就是 batch × action_horizon × action_dim。
- 具体尺寸由各自 config 定死。Pi 在 pi0_config.py 和它的 pi0_config.py 里定义；ACoT-VLA 在 acot_vla.py 和 acot_vla.py 里定义。两者的观测输入规格基本一致：三路图像、一个 state 向量、一个 tokenized prompt，最终动作规格都是 action_horizon × action_dim。
- ACoT 比普通 Pi 多了一条训练监督 “coarse_actions”。这条不是走公共 model.py 暴露出来的，而是直接在 ACOT_VLA.compute_loss 和 train.py 里单独传入。

**实现层**
- 原始环境数据先经过 transforms 和 policy 封装，再变成 Observation。总入口在 policy_config.py 和 Policy.infer。
- 普通 Pi 的输入变换由 config.py 负责，里面会做默认 prompt 注入、图像 resize、prompt tokenize、state 和 actions 对齐。关键变换类是 transforms.py、transforms.py、transforms.py 和 transforms.py。
- 如果是 Pi0-FAST，这条支路在 config.py，输入输出会走 token 化和反解码，对应 transforms.py 和 transforms.py。
- 以 SRB 为例，外部观测怎么变成模型输入，定义在 srb_policy.py；模型输出怎么裁成环境动作，定义在 srb_policy.py。

**Pi 模型**
- Pi 本体在 pi0.py。
- pi0.py 负责把多路图像编码成视觉 tokens，把语言 prompt 编成文本 tokens，然后拼成前缀。
- pi0.py 负责把 state、加噪后的 actions、timestep 编成动作 expert tokens。
- pi0.py 用的是 flow matching：先把真实动作加噪得到 x_t，再让模型预测速度场 v_t，去逼近目标 u_t = noise - actions。
- pi0.py 则从高斯噪声开始，按时间步迭代去噪，最后直接产出一段动作序列。

**ACoT-VLA 模型**
- ACoT 本体在 acot_vla.py。
- 它的前缀编码和 Pi 类似，还是看 acot_vla.py；真正差异在 acot_vla.py。
- 这里 suffix 分成两种模式：“reasoner” 先建模 coarse trajectory，“expert” 再建模最终动作。显式 reasoning 会把 coarse trajectory 当作显式动作意图；隐式 reasoning 会从 prefix 的 KV cache 抽特征，再和 expert tokens 做融合。
- acot_vla.py 里会同时训练 coarse_actions 和 actions 两条流；acot_vla.py 里会先采样 coarse_actions，再条件化采样最终 actions。
- 所以 ACoT 的输出不是永远只有一条动作。显式 reasoning 打开时，它会同时返回 acot_vla.py；否则只返回 acot_vla.py。

**你现在看代码时要注意的一点**
- 这个仓库里 ACoT 的公共接口还没有完全收口。公共 model.py 只声明了 PI0、PI0_FAST、PI05，但 ACOTConfig.model_type 直接返回 ACoT 专用枚举值；同时通用 Policy.infer 还是按“顶层只有 actions”去封装，而 ACoT 的 sample_actions 可能返回一个字典。所以看 ACoT 时，最稳的路径不是只看通用接口，而是直接沿 acot_vla.py 和具体环境的 ACoT adapter 看，比如 arx_policy.py 和 arx_policy.py。
- SRB 这条分支目前也明确写了，接的是标准 VLA 路径，不是 ACoT 的 coarse_actions 三路训练路径，见 README.md 和 README.md。

如果你要，我下一步可以继续做两件事里的一个：
1. 顺着 serve_policy.py 这条 ACoT 默认入口，把从配置、数据变换到 sample_actions 的完整调用链画出来。
2. 按一个具体 batch，把 ACoT 和 Pi 的每个输入字段、输出字段和 shape 一项项列出来。


Optimized tool selection我继续把两件事都补全：先把 ACoT 默认服务入口的真实调用链串起来，再用具体 batch 视角把 Pi 和 ACoT 的输入输出字段与 shape 列清楚。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/scripts/serve_policy.py)

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/data_loader.py#1-1), lines 1 to 140

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/config.py#1-1), lines 1 to 120

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/model_test.py)

我补一个关键核对：确认 ACoT 配置名到底有没有注册进 get_config，这决定默认服务入口是“完整可跑”还是“脚本只写了占位名”。

Ran terminal command:  grep -RIn "name="acot\|acot_icra_simulation_challenge_reasoning_to_action\|acot_libero_action_cot_explicit_implicit_co_fusion" src scripts

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/models/pi0_fast.py)

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/policies/vlabench_policy.py#90-90), lines 90 to 157

我再把服务端的请求链和训练时 batch 是怎么拼出来的补上，这样 1 和 2 就能闭环。

Searched for regex `create_data_loader|Observation.from_dict|yield|actions|coarse_actions` (`**/src/openpi/training/data_loader.py`), 13 results

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/data_loader.py#140-140), lines 140 to 360

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/serving/websocket_policy_server.py)

我再把 data loader 的收口部分读出来，确认训练 batch 最终到底是二元组还是三元组。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/data_loader.py#360-360), lines 360 to 541

Searched for regex `class DataLoaderImpl|yield _model\.Observation\.from_dict\(batch\), batch\["actions"\]` (`**/src/openpi/training/data_loader.py`), 2 results

Searched for regex `yield _model\.Observation\.from_dict\(batch\), batch\["actions"\], batch\["coarse_actions"\]` (`**/src/openpi/training/data_loader.py`), no results

我把这次确认到的两个关键事实记进仓库记忆里：默认 ACoT 配置名在当前源码里找不到注册，且通用 data loader / policy 接口仍偏向单路 actions。

Inserted into memory file [](file:///home/omnisky/.config/Code/User/workspaceStorage/f3b3bbf6136ff0068835e337cafa05d5/GitHub.copilot-chat/memory-tool/memories/repo/ACoT-VLA_saa.md)

我补一组函数入口的精确定位，回答里会直接按这条链路展开。

Searched for regex `def create_default_policy|def create_policy|def main\(` (`**/scripts/serve_policy.py`), 3 results

Searched for regex `def __iter__\(|class DataLoaderImpl` (`**/src/openpi/training/data_loader.py`), 7 results

Searched for regex `def _handler|class WebsocketPolicyServer` (`**/src/openpi/serving/websocket_policy_server.py`), 2 results

这次我把两件事都补齐了：一是把 ACoT 默认服务入口从脚本一路追到模型采样；二是按具体 batch 视角把 Pi 和 ACoT-VLA 的输入输出字段与 shape 摊开。结论先说在前面：当前仓库里，ACoT 的模型本体实现是有的，但默认服务、训练 data loader 和通用 policy 封装这几层还没有完全对齐，所以这条链更像“目标调用路径”，不是当前树上已经打通的闭环。

**1. 调用链**
1. 默认 ACoT 服务入口在 serve_policy.py。这里把 G2SIM 的默认 checkpoint 绑定到配置名 acot_icra_simulation_challenge_reasoning_to_action。
2. 启动流程是 serve_policy.py -> serve_policy.py -> serve_policy.py。核心动作只有两步：先用 get_config 取 TrainConfig，再交给 policy_config.py。
3. create_trained_policy 会加载 checkpoint，然后按固定顺序组装推理前后的变换链。输入链是 repack transforms -> data transforms -> transforms.py -> model transforms；输出链是 model output transforms -> transforms.py -> data output transforms。模型相关的输入变换由 config.py 定义，常用的几步是 transforms.py、transforms.py，FAST 分支还会用 transforms.py 和 transforms.py。
4. 服务端收到 websocket 请求后，真正落到 WebsocketPolicyServer._handler。这里会把客户端发来的 observation 解包，然后直接调用 Policy.infer。
5. Policy.infer 做三件事：先把单条 observation 批量化成 batch=1；再调用 Observation.from_dict 转成模型结构体；最后执行模型的 sample_actions，并把输出再过一遍 output transforms。
6. 如果走普通 Pi 路径，最终进入的是 Pi0.sample_actions。这条路先做 prefix 编码，再在动作空间里从高斯噪声迭代去噪，最后返回一段连续动作序列。
7. 如果走 ACoT-VLA 路径，目标终点是 ACOT_VLA.sample_actions。这条路会先编码 prefix，再可选地从 KV cache 抽隐式 action reasoning，之后先采样 coarse trajectory，再条件化采样 fine actions；显式 reasoning 打开时，返回值是 acot_vla.py，否则只有 acot_vla.py。

当前树上的实际断点有 5 个，都是静态代码里能直接看到的：
- 默认 ACoT 配置名能在 serve_policy.py 找到，但我在当前 src 和 scripts 里没有搜到对应的 TrainConfig 注册；而 get_config 的实现就在 config.py。
- config.py 只写了 PI0、PI05、PI0_FAST 三个分支，没有 ACoT 分支。
- 公共 model.py 只有 PI0、PI0_FAST、PI05，但 ACOTConfig.model_type 会返回 ACOT_VLA_PI0 和 ACOT_VLA_PI05。
- 通用训练 data loader 的最终收口在 DataLoaderImpl.__iter__，它只 yield Observation 和 actions，实际返回在 data_loader.py；但 ACoT 训练步 train.py 需要的是 Observation、actions、coarse_actions 三元组。
- 通用 Policy.infer 在 policy.py 把模型输出包成一个顶层 actions 字段；而 ACoT 的 sample_actions 本身已经可能返回一个带 coarse_actions 的字典，这两边接口现在并不一致。

所以，按设计意图，这条链应该是：
serve_policy.py -> policy_config.py -> policy.py -> acot_vla.py

但按当前源码状态，这条链在配置注册、模型类型、训练 batch 和推理返回值四个层面都还差最后一段胶水代码。

**2. 具体 batch 的输入输出和 shape**
下面我按“模型内部看到的 batch”来讲。这里不假设那个缺失的 acot 配置名已注册，而是直接用类本身的默认规格。

先看公共结构。所有标准 Pi / ACoT 观测都先归一成 model.py，字段是 image、image_mask、state、tokenized_prompt、tokenized_prompt_mask；动作张量公共别名在 model.py，shape 是 B × action_horizon × action_dim。

Pi0 / Pi05 的规格来自 Pi0Config.inputs_spec：
- image.base_0_rgb: B × 224 × 224 × 3
- image.left_wrist_0_rgb: B × 224 × 224 × 3
- image.right_wrist_0_rgb: B × 224 × 224 × 3
- image_mask 对应每路图像: B
- state: B × action_dim，默认是 B × 32
- tokenized_prompt: B × max_token_len，Pi0 默认是 B × 48，Pi05 默认是 B × 200，默认值定义在 pi0_config.py
- tokenized_prompt_mask: B × max_token_len
- 训练目标 actions: B × action_horizon × action_dim，默认是 B × 50 × 32

Pi 的实现细节是：
- pi0.py 处理图像 token 和语言 token。
- pi0.py 处理 state、加噪 actions、timestep。
- pi0.py 输出的是 B × action_horizon，也就是默认 B × 50。
- pi0.py 输出的是 B × 50 × 32。

如果把它放到单条推理样本里看，经过 Policy.infer 的 batchify 之后，模型实际看到的是：
- state: 1 × 32
- 每张图像: 1 × 224 × 224 × 3
- tokenized_prompt: 1 × 48 或 1 × 200
- 最终 sample_actions 输出: 1 × 50 × 32
- infer 结束后再去掉 batch 维，客户端拿到的是 50 × 32

ACoT-VLA 的观测规格来自 ACOTConfig.inputs_spec，观测字段和 Pi 基本同型，但默认超参数不同，定义在 acot_vla.py：
- state: B × 32
- 三路图像: B × 224 × 224 × 3
- tokenized_prompt: 因为 pi05 默认打开，所以默认是 B × 200
- fine actions 目标: B × 30 × 32
- 额外的 coarse_actions 监督: B × 50 × 32。这个不是 inputs_spec 的返回值，而是 acot_vla.py 额外要求的第三个训练输入。

ACoT 的实现细节是：
- acot_vla.py 仍然负责图像和语言前缀。
- acot_vla.py 分成 reasoner 和 expert 两种后缀模式。
- acot_vla.py 同时对 coarse_actions 和 fine actions 做 flow matching。
- acot_vla.py 先采样 coarse，再采样 fine。

如果按单条推理样本看，ACoT 理想状态下模型内部会看到：
- state: 1 × 32
- 每张图像: 1 × 224 × 224 × 3
- tokenized_prompt: 1 × 200
- coarse branch 的噪声和轨迹张量: 1 × 50 × 32
- fine branch 的噪声和轨迹张量: 1 × 30 × 32
- 显式 reasoning 打开时，sample_actions 输出应是 coarse_actions: 1 × 50 × 32，加上 actions: 1 × 30 × 32
- 显式 reasoning 关闭时，只返回 actions: 1 × 30 × 32

这里还有一个你在看 batch 时要特别注意的点：环境适配层经常会再裁掉动作维度。比如 VLABench 的 ACoT 适配器在 vlabench_policy.py 里会从原始动作序列拆出 coarse_actions 和 actions，在 vlabench_policy.py 里只把前 7 维动作返回给环境。所以“模型内部 action_dim=32”和“环境最终拿到 7 维或 14 维动作”是两个不同层次。

和你当前 SRB 相关的补充结论是：
- SRB 这条接入在 srb_policy.py 和 config.py 走的是标准 VLA 输入输出封装。
- 它目前不是 ACoT 的 coarse_actions 三路训练路径，这点在 README.md 和 README.md 也写明了。

补一条，如果你说的 pi 还包括 PI0-FAST：
- 模型级 sample_actions 在测试里返回的是 token 序列，不是直接的连续动作，见 model_test.py 和 pi0_fast.py。
- 这条路会在 output transforms 里通过 transforms.py 把 token 反解成 action_horizon × action_dim。

以上结论都是基于当前源码静态追踪，没有直接跑通 G2SIM 那条默认 ACoT 服务入口；原因一是当前树里没找到对应的 ACoT TrainConfig 注册，原因二是你这个环境本身还有 torch 和 torchvision 的 CUDA 版本不匹配问题。

1. 如果你要，我下一步可以直接把这几处 ACoT 断点补成一条真正可跑的服务链。
2. 如果你更关心调试，我可以按你当前 SRB 的 observation 字典，给你画一份从原始输入到最终 action chunk 的逐字段映射表。

## norm_states

Optimized tool selection我先快速看一下 compute_norm_stats.py 的实现和它调用的训练配置入口，然后给你一个直接可用的用途说明。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/scripts/compute_norm_stats.py)

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/config.py#1-1), lines 1 to 260

我已经确认这个脚本会按配置读取数据并统计归一化参数。下一步我看一下归一化文件的保存格式和训练时的读取路径，给你完整闭环说明。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/shared/normalize.py)

Searched for regex `compute_norm_stats.py|norm_stats|normalize.load|normalize.save` (`**/src/**`), 86 results

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/transforms.py#1-1), lines 1 to 360

我已经定位到训练入口对这个脚本的硬依赖提示了。再补一段数据加载代码后，我就给你结论和典型使用时机。

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/data_loader.py#150-150), lines 150 to 240

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/checkpoints.py#50-50), lines 50 to 140

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/policies/policy_config.py)

Searched for regex `assets_dirs|class TrainConfig|asset_id` (`**/src/openpi/training/config.py`), 42 results

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/src/openpi/training/config.py#500-500), lines 500 to 620

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/docs/norm_stats.md)

Searched for regex `def main|get_config|create_rlds_dataloader|create_torch_dataloader|keys =|sample_ratio|normalize.save|output_path|tyro.cli|RemoveStrings|rlds_data_dir` (`**/scripts/compute_norm_stats.py`), 18 results

Searched for text `Normalization stats not found|Make sure to run` (`**/src/openpi/training/data_loader.py`), 4 results

Searched for regex `def _load_norm_stats|_normalize.load|data_assets_dir|norm_stats=self._load_norm_stats` (`**/src/openpi/training/config.py`), 7 results

Searched for regex `assets_base_dir|def assets_dirs|return \(pathlib.Path\(self.assets_base_dir\) / self.name\)` (`**/src/openpi/training/config.py`), 3 results

Searched for regex `def save\(|def load\(|norm_stats.json` (`**/src/openpi/shared/normalize.py`), 4 results

Searched for regex `class Normalize|return \(x - mean\) / \(std \+ 1e-6\)|_normalize_quantile` (`**/src/openpi/transforms.py`), 4 results

Searched for regex `use_quantile_norm=model_config.model_type != ModelType.PI0|use_quantile_norm: bool` (`**/src/openpi/training/config.py`), 2 results

Searched for regex `transforms.Normalize|transforms.Unnormalize|load_norm_stats` (`**/src/openpi/policies/policy_config.py`), 3 results

相关链路已经查清了：这个脚本就是训练前生成归一化统计文件的步骤。现在我整理成一句话结论 + 关键细节和注意点给你。

Searched for text `compute_norm_stats.py --config-name` (`**/README.md`), 2 results

Read [](file:///media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/examples/srb/README.md#60-60), lines 60 to 130

我再快速看一下仓库记忆里的已有结论，避免给你和项目既有约定冲突的说明。

Read memory [](file:///home/omnisky/.config/Code/User/workspaceStorage/f3b3bbf6136ff0068835e337cafa05d5/GitHub.copilot-chat/memory-tool/memories/repo/ACoT-VLA_saa.md)

Inserted into memory file [](file:///home/omnisky/.config/Code/User/workspaceStorage/f3b3bbf6136ff0068835e337cafa05d5/GitHub.copilot-chat/memory-tool/memories/repo/ACoT-VLA_saa.md)

compute_norm_stats.py 的作用是训练前生成归一化统计参数。它会按配置加载数据，统计 state、actions、coarse_actions 的分布（均值、标准差和分位数），然后写出 norm_stats.json，供后续数据归一化使用。compute_norm_stats.py compute_norm_stats.py compute_norm_stats.py normalize.py

这一步基本是训练前置条件，因为训练数据管线会调用 Normalize；如果没有统计文件，会直接报错并提示先跑这个脚本。data_loader.py data_loader.py transforms.py transforms.py

另外一个实用提醒：当前脚本把结果写到当前目录，而配置自动加载默认会去 assets_base_dir/config_name/asset_id 下找。跑完还报缺失时，常见原因就是文件路径不一致。compute_norm_stats.py config.py config.py config.py
