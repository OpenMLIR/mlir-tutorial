MLIR（Multi-Level Intermediate Representation）包含了更多的信息，抽象层级更高。我在2023.8开始从事MLIR编译器的开发，当时学习资料不多，基本都是手撕其他项目的源码（当然现在也是）。2023.11我把这些开源项目整理了下发表在我的博客里，现在挪到知乎来。

对MLIR不了解可以看下我的这篇文章。
[从零开始教你写一个MLIR Pass](https://zhuanlan.zhihu.com/p/708819963)

# 一、OpenAI

## 1、[triton-lang/Triton 16k](https://github.com/triton-lang/triton)

Pytorch 2.0是史诗级更新，OpenAI的Triton也随着chatgpt的爆火而备受关注。带来了`Triton`和`Triton GPU`高层次抽象的dialect，能够很好的表示GPU的硬件细节，Pytorch 2.0[更新日志](https://pytorch.org/get-started/pytorch-2.0/#developervendor-experience)说`Triton`有95%的库水平。

![](https://img2023.cnblogs.com/blog/1154439/202312/1154439-20231206133230428-833031751.png)

2025年6月我也基于[pytorch-labs/tritonbench](https://github.com/pytorch-labs/tritonbench) 做了性能对比，如下图所示，可以看到非常快，更多对比见[triton是否会冲击cuda生态？ - BobHuang的回答](https://www.zhihu.com/question/1919763006750975400/answer/1921121681612739823)。

![](https://picx.zhimg.com/v2-745a86a33b7205ed63eb85b26dff61a8_r.jpg?source=c8b7c179)

我写了一篇 [浅析 Triton 执行流程](https://www.cnblogs.com/BobHuang/p/18324040)，有兴趣可以看看。Triton是目前最成功的MLIR项目。

seed最近开源了分布式的Triton [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)，激动人心。

除了[nvidia](https://github.com/triton-lang/triton/tree/main/third_party/nvidia)、[amd](https://github.com/triton-lang/triton/tree/main/third_party/amd)、[intel](https://github.com/intel/intel-xpu-backend-for-triton/tree/main/third_party/intel)、[cpu](https://github.com/triton-lang/triton-cpu)、[microsoft(部分)](https://github.com/microsoft/triton-shared)、[meta tlx扩展(Low-level)](https://github.com/facebookexperimental/triton/tree/tlx/third_party/tlx)这些后端外，我们还有[昆仑芯](https://github.com/FlagTree/flagtree/tree/main/third_party/xpu)、[摩尔线程](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads)、[沐曦](https://github.com/FlagTree/flagtree/tree/main/third_party/metax)、[ARM china](https://github.com/FlagTreeZhouyi/flagtree-zhouyi/tree/master/third_party/aipu)、[华为昇腾](https://gitee.com/ascend/triton-ascend/tree/master/ascend)、[清微智能](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro)、[天数智芯](https://github.com/FlagTree/flagtree/tree/main/third_party/iluvatar)、[寒武纪(部分)](https://github.com/Cambricon/triton-linalg)、[Seed 分布式扩展Op](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/dialect/include/Dialect/Distributed/IR/DistributedDialect.td) 的后端实现可以参考。

搞个DSL 再lower到Triton你又该如何应对，好好好，支持[helion](https://github.com/pytorch-labs/helion)

# 二、LLVM

## 1、[官方的torch前端torch-mlir 1.6k](https://github.com/llvm/torch-mlir)

可以将torch的model转换到MLIR中。

## 2、[官方在开发的clangir 0.51k](https://github.com/llvm/clangir)

这个项目我觉得蛮有意思的，clang但是以MLIR作为IR。`cgeist`在翻译时用的还是`clang`，因为AST的信息和MLIR不对等所以有些东西不太好表示。期待早日做好将大大提高`clang`的表达能力。我认为还是有实现前端**大一统**的潜力。目前进度是“可以处理整个C和一部分C++”，affine有人在做。

## 3、[官方的C/C++前端Polygeist 0.56k](https://github.com/llvm/polygeist)

可以将C/C++代码翻译到`affine`层级，工具名称是`cgeist`。有时候不太会写输出结果，可以参考下他的输出。最新版的Polygeist增加了`poly`，你也可以学习下poly的接入方式。

## 4、[LLVM中的toy 33.3k](https://github.com/llvm/llvm-project/tree/main/mlir/examples/toy)
适合上手学习调试，官方还有[文档](https://mlir.llvm.org/docs/Tutorials/Toy)

## 5、[LLVM中的Fortran前端Flang 33.k](https://github.com/llvm/llvm-project/tree/main/flang)
LLVM中的Fortran前端，这个也做得蛮早了，但是还不够完善。最初是[f18 project](https://github.com/flang-compiler/f18)

# 三、Google

## 1、[iree-org/iree 3.2k](https://github.com/iree-org/iree)

MLIR大模型推理，很多参考了这个项目。做推理框架首先看这个，当然推理框架还有[llama.cpp](https://github.com/ggml-org/llama.cpp)、[ollama](https://github.com/ollama/ollama)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[vllm](https://github.com/vllm-project/vllm)、[sglang](https://github.com/sgl-project/sglang)。

OpenXLA 是 Google 想将编译器相关技术从 TensorFlow 独立出来的项目组，里面有hlo dialect的定义，fusion pass的思路。这个貌似着工业界用的还不少，我以前以为处于Pytorch的大一统时代，但是部署都糊想办法省钱的。现在IREE很多也都转vllm了，毕竟要追求性能。

IREE里隐藏着对 [StableHLO](https://github.com/openxla/stablehlo)，这是定义的神经网络模型的高层级运算。
当然XLA也是可以接进Pytorch的，有[Pytorch/XLA](https://github.com/pytorch/xla)。

IREE和TVM的对比
![](https://img2024.cnblogs.com/blog/1154439/202403/1154439-20240307093101922-162675036.png)来自[Compiler Technologies in Deep Learning Co-Design: A Survey](https://spj.science.org/doi/epdf/10.34133/icomputing.0040)

## 2、[openxla/stablehlo 0.50k](https://github.com/openxla/stablehlo)

## 3、[LLVM中的mlir 33.3k](https://github.com/llvm/llvm-project/tree/main/mlir)

`Chris Lattner`2018年在`Google Brain`主导开发了`MLIR`，并在2019年4月正式开源出来。Chris Lattner在[AI民主化](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls)系列文章对MLIR的中肯评价如下。

> As I worked to scale Google TPUs in 2017 and 2018, a pattern emerged: first-generation AI frameworks like TensorFlow and PyTorch lacked scalability, while the second generation AI compilers like XLAsacrificed flexibility. To break this cycle, I led the team to build a new MLIR compiler framework—a modular, extensible compiler framework designed to support AI’s rapidly evolving hardware landscape.

> Did it succeed? MLIR drove industry-wide breakthroughs—Python DSLs like Triton, cuTile, and others were built on top of it, redefining GPU programming. But like TVM and XLA before it, MLIR faces governance challenges, fragmentation, and competing corporate interests. The vision of a truly unified AI compiler stack still seems just out of reach, caught in the same power struggles that have shaped the industry for decades.

# 四、AMD

都说AMD在梭哈MLIR实现弯道超车，具体怎么样可以看代码。Nod.ai的人还在搞新的[dsl](https://github.com/llvm/eudsl)，Nod.ai、Xilinx都被AMD收购了。

## 1、[AMD rocMLIR](https://github.com/ROCm/rocMLIR)

可以针对AMD硬件做CONV和GEMM kernel生成，被[MIGraphX](https://github.com/ROCm/AMDMIGraphX)使用。他们还实现了xmir-runner，如果有自制MLIR JIT runner可以参考下。

## 2、[Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)
An MLIR-based toolchain for AMD AI Engine-enabled devices.

## 3、[Xilinx AIR platforms](https://github.com/Xilinx/mlir-air)

Xilinx是做FPGA的，被AMD收购了，FPGA中也有AI引擎，他们梭哈MLIR蛮久了，开源质量不错。
大多数都是转换类，[Pass位置](https://github.com/Xilinx/mlir-air/tree/main/mlir/lib/Transform)
这个还能看到有人在实际运用，比如[mase](https://github.com/DeepWok/mase)，有100多个star，不过HLS基本都是国人在搞


## 4、[Xilinx/onnx-mlir](https://github.com/Xilinx/onnx-mlir)

# 五、NVIDIA

## 1、[NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)

只开源了前端非常少的一部分，但是我们可以看到MLIR的生态大家都在做。很大的一块收益是编译速度的显著提升，你不用在编译CUTLASS时等待很久。提升>100x，在去掉C++模版类实例化使用MLIR后，8kx8kx8k GEMM从C++编译时间的27997ms来到了241ms，十分惊人。
![](https://img2024.cnblogs.com/blog/1154439/202506/1154439-20250619053717517-1393726606.png#w80)

带来的收益还有Pytorch的集成，这点对不熟悉cutlass C++但是想来写算子的人是非常**友好**的。

# 六、intel

## 1、[intel/graph-compiler](https://github.com/intel/graph-compiler)
这个MLIR接的是[oneDNN Graph API](https://oneapi-src.github.io/oneDNN/graph_extension.html)，也是最近刚开始做

## 2、[intel/intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)

intel的xpu后端，这里也有和微软[triton-shared](https://github.com/microsoft/triton-shared)类似的部分。

# 七、微软

## 1、[microsoft/triton-shared](https://github.com/microsoft/triton-shared)

微软最先在linalg上做了一些尝试，并把自己的一些优化share出来了。[This talk at the 2023 Triton Developer Conferene](https://www.youtube.com/watch?v=y2V3ucS1pfQ)

# 八、meta

META可以看到的是在折腾Triton，[Meta 推荐芯片MTIAV2](https://zhuanlan.zhihu.com/p/1922926865947006451) 也使用了Triton这套软件栈。
![image](https://img2024.cnblogs.com/blog/1154439/202507/1154439-20250704194235283-548905498.png)

## 1、[facebookexperimental/triton](https://github.com/facebookexperimental/triton/tree/tlx/third_party/tlx)
meta在搞TLX (Triton Low-level Language Extensions)，把 warp-aware, hardware-near 带回Triton，以求拿到性能。把 Low-level 带回Triton也是有收益的，能拿到性能。

## 2、[pytorch-labs/triton-cpu](https://github.com/pytorch-labs/triton-cpu)
也折腾过Triton的cpu后端的。

# 九、ONNX

## 1、[onnx/onnx-mlir](https://github.com/onnx/onnx-mlir)
这个项目我还真用过，要把ONNX转换到MLIR dialect，不少krnl的需要自己实现。所以经常是一边看netron可视化工具，一边对语义，写实现。


# 九、Jim Keller的tenstorrent

## 1、[tenstorrent tt-mlir](https://github.com/tenstorrent/tt-mlir)
tenstorrent是Jim Keller(硅仙人)领导的AI芯片新创公司，目标也是他们的AI加速器。兆松(1nfinite)有一个中文[文档](https://zhuanlan.zhihu.com/p/717019727)

# 十、硅基流动

## 1、[Oneflow-Inc/oneflow](https://github.com/Oneflow-Inc/oneflow)
一流科技较早将MLIR融入自己的深度学习框架，也对MLIR做了些支持。不过在深度学习中我觉得很多依旧是Transform类的，能lower正确就可以，很像一个工程问题。[定义创建Pass的文件](https://github.com/search?q=repo%3AOneflow-Inc%2Foneflow+std%3A%3Aunique_ptr%3Cmlir%3A%3APass%3E&type=code&p=1)，甚至还有PDLL这类高级功能。一流科技也就是现在的硅基流动。

# 十一、算能

## 1、[sophgo/tpu-mlir](https://github.com/sophgo/tpu-mlir)
TPU-MLIR是一个面向深度学习处理器的开源TPU编译器。该项目提供了完整的工具链，将各种框架下预训练的神经网络转换为可在TPU中高效运行的二进制文件bmodel，以实现更高效的推理。TPU-MLIR项目已应用于算能开发的最新一代深度学习处理器BM1684X。结合处理器本身的高性能ARM内核以及相应的SDK，可以实现深度学习算法的快速部署。甚至提供了[相应课程](https://www.sophgo.com/curriculum/description.html?category_id=9)，只需要填写相关信息就可以学习，B站不用注册，来白嫖[B站这边](https://www.bilibili.com/video/BV1yP4y1d7gz)。

# 十二、软件所iscas

## 1、[buddy-mlir](https://github.com/buddy-compiler/buddy-mlir)
适合上手学习调试，不过已经越来越强大了。其在最近（2023年11月）完成了端到端 LLaMA2-7B 推理示例

# 十三、华为

## 1、[mindspore-ai/akg](https://github.com/mindspore-ai/akg/tree/r2.3)

AKG是Auto Kernel Generator的简称，在昇思MindSpore框架中担任图算融合编译加速的任务。AKG基于polyhedral多面体编译技术，可自动生成满足并行性与数据局部性的调度，目前能够支持NPU/GPU/CPU等硬件。

MindSpore AKG MLIR已经支持主流模型中所有重要算子，但是随着网络的迭代和算法的改进，依然不断有新的算子出现。由于MindSpore AKG支持包括NPU、GPU、CPU在内的多硬件后端，我们规划了基于后端代码生成能力完成相关算子支持，包括：1.对于新算子提供基于循环和数学表达式的表达。2.对于融合算子提供以已有算子拼接的展开表达。

B站有对应的[技术分享会录播](https://www.bilibili.com/video/BV1V4WheFEQd)

源码大概看了下，为了解决动态shape弄了个symbolicStrExprMap，要SameSymbolicShape。用的是symengine，而不是MLIR的affine dialect

GPU Lower流程如下所示
![](https://img2024.cnblogs.com/blog/1154439/202504/1154439-20250417072435773-966281194.png)

发布后就没怎么更新了，心痛，想学。

# 十四、字节跳动

## 1、[bytedance/byteir](https://github.com/bytedance/byteir)
ByteIR项目是字节跳动的模型编译解决方案。ByteIR包括编译器、运行时和前端，并提供端到端的模型编译解决方案。 尽管所有的ByteIR组件（编译器/运行时/前端）一起提供端到端的解决方案，并且都在同一个代码库下，但每个组件在技术上都可以独立运行。

# 十五、阿里

## 1、[alibaba/BladeDISC](https://github.com/alibaba/BladeDISC)

阿里云PAI团队的BladeDISC目的是解决AI编译器的Dynamic Shape问题，主要做了显存优化、计算优化、通信优化，2024年底发现其[推文](https://mp.weixin.qq.com/s/BHf-5RNDpWC9654HVwSTZw)多了起来，这个工作的[论文](https://arxiv.org/abs/2412.16985)被 NeurIPS Workshop 2024收录了。


# 十六、寒武纪

## 1、[Cambricon/triton-linalg](https://github.com/Cambricon/triton-linalg)

寒武纪针对mlu的到linalg的实现，linalg之后和硬件实现有关。这是发布时的[Slides，72页开始](https://baai.org/l/FHcyY)
![](https://img2024.cnblogs.com/blog/1154439/202406/1154439-20240615171535010-1100495202.png)


# 十七、一些教程

## 1、[OpenMLIR/mlir-tutorial](https://github.com/OpenMLIR/mlir-tutorial)
北大周可行的中文教程，更适合中国宝宝使用。我完善了两章，之后还想加点新东西。

## 2、[KEKE046/mlir-tutorial](https://github.com/KEKE046/mlir-tutorial)
北大周可行的中文教程原地址

## 3、[j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial)
Jeremy Kun的英文教程

## 4、[BBuf/tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn)
BBuf的学习笔记
