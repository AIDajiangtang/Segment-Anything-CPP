English | [简体中文](ReadmeCN.md)<br/>
In the SAMTool-CSharp repository, we use the C# language and the ONNXRuntime for.Net framework to perform inference on the visual large-scale model "Segment Anything", and use WPF to interact with users and display segmentation results.<br/>

In the SAMTool-CPP repository, we will use the C++ language and the ONNXRuntime for CPP framework to perform inference on the visual large-scale model "Segment Anything". This is done for two reasons: first, to improve running efficiency; and second, to have access to source code-level platforms.<br/>

This way, you can deploy the model to embedded devices such as Windows, Linux, and even Android. The UI is not limited to WPF either - you can choose any option between QT, Html, and Winform, and we will provide a version for WPF.<br/>

SAM supports Promots including Point, Box, Mask, and Text. The first three are relatively easy to implement and understand, while Text Promot is relatively complex.<br/>

For Text Promot, it is necessary to introduce another deep learning model, CLIP. Simply put, it can map text and images to the same vector space, so that text and images can be compared.<br/>

First, SAM is used to segment all targets, and then a image is cropped around each target and the image is input to CLIP to calculate the image embedding.<br/>

Then, the input text embedding is calculated through CLIP. Finally, the cosine similarity between the text embedding and all target image embeddings is calculated to find the target image that is most similar to the text.<br/>
