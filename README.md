简体中文 | [English](ReadmeEN.md)  

# segment anything（SAM）  
[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`源码`](https://github.com/facebookresearch/segment-anything/)]  

# Ours：segment anything（SAM） for CPP Inference  
在SAMTool-CSharp仓库中，我们使用C#语言，ONNXRuntime for.Net框架对视觉大模型Segment Anaything完成了推理过程，并使用WPF与用户交互以及显示分割结果。  
在本仓库SAMTool-CPP中，我们将使用C++语言，ONNXRuntime for CPP框架对视觉大模型Segment Anaything完成了推理过程，这样做出于两点考虑，第一，提升运行效率，第二，源码级款平台。  
这样你就可以将模型部署到Windows，Linux，甚至Android等嵌入式设备中。  
UI也不仅限于WPF了，你可以在QT，Html，Winform中任意选择，我们会提供WPF的版本。  

# Text Promot：CLIP  
SAM支持的promot有Point，Box，Mask以及Text文本，前三个相对比较容易实现和理解，Text Promot相对比较复杂。  
对于Text Promot，需要引入另一个深度学习模型CLIP，简单来说，他能够将文本和图像映射到同一个向量空间中，这样文本和图像就能够进行比较了。   
首先通过SAM分割出所有的目标，然后以每个目标为中心裁剪出一个图像，并将这个图像输入到CLIP中计算图像Embedding  
然后通过CLIP计算输入文本的Embedding，最后计算文本Embedding和所有目标图像Embedding的余弦相似度，找到与文本最相似的目标图像。   
