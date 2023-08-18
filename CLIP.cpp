#include"CLIP.h"
#include"opencv2/imgproc.hpp"

CLIP::CLIP()
{
    std::wstring encoderwstr = L"D:\\visual.onnx";
    std::wstring decoderwstr = L"D:\\textual.onnx";
    this->LoadONNXModel(L"D:\\visual.onnx", L"D:\\textual.onnx");

    this->mTokenlizer = std::unique_ptr<SimpleTokenizer>(new SimpleTokenizer());
}
void CLIP::LoadONNXModel(std::wstring visualpath, std::wstring textualpath)
{
    this->m_env = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CLIP"));
    this->m_sessionOption = std::unique_ptr<Ort::SessionOptions>(new Ort::SessionOptions());
    this->m_sessionOption->SetIntraOpNumThreads(1);
    this->m_sessionOption->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    this->m_ImgEncoder = std::unique_ptr<Ort::Session>(new  Ort::Session(*this->m_env.get(), visualpath.c_str(), *this->m_sessionOption.get()));
    this->m_TxtEncoder = std::unique_ptr<Ort::Session>(new  Ort::Session(*this->m_env.get(), textualpath.c_str(), *this->m_sessionOption.get()));
}
void CLIP::ImgEncoder(cv::Mat img, std::vector<float>& embedding, std::vector<int64_t>& embeddingshape)
{
    //cv::Mat img = cv::imread("D:\\CLIP.png");
    cv::Mat resizeImage;
    cv::resize(img, resizeImage, cv::Size(224, 224), cv::INTER_AREA);
    resizeImage.convertTo(resizeImage, CV_32F);
    std::vector<float>imgdata; imgdata.resize(3 * 224 * 224);

    for (int i = 0; i < resizeImage.rows; i++) {
        for (int j = 0; j < resizeImage.cols; j++) {
            float b = resizeImage.at<cv::Vec3f>(i, j)[0];
            float g = resizeImage.at<cv::Vec3f>(i, j)[1];
            float r = resizeImage.at<cv::Vec3f>(i, j)[2];
            int index = i * resizeImage.cols + j;
            imgdata[index] = r;
            imgdata[224 * 224 + index] = r;
            imgdata[224 * 224 * 2 + index] = r;
        }
    }

    const char* inputNames[] = { "input" }, * outputNames[] = { "output" };
    vector<int64_t> imgShape{ 1, 3, 224, 224 };
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 构造ONNXRuntime的OrtValue对象
    Ort::Value imgTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgdata.data(), imgdata.size(), imgShape.data(), imgShape.size());
    Ort::RunOptions run_options;
    vector<Ort::Value> Outputs = this->m_ImgEncoder->Run(run_options, inputNames, &imgTensor, 1, outputNames, 1);

    std::vector<int64_t>OutShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* imgembedding = (float*)Outputs[0].GetTensorMutableData<void>();

    embeddingshape.resize(OutShape.size());
    int totalsize = 1;
    for (int i=0;i< OutShape.size();i++)
    {
        totalsize *= OutShape[i];
        embeddingshape[i] = OutShape[i];
    }
    embedding.resize(totalsize);
    for (int i = 0; i < totalsize; i++)
    {
        embedding[i] = imgembedding[i];
    }

}
void CLIP::TxtEncoder(std::wstring txt, vector<int64_t>txttokenShape,std::vector<float>& embedding, std::vector<int64_t>& embeddingshape)
{
    std::vector<int64>txtToken = this->mTokenlizer->tokenlize(txt);

    const char* inputNames[] = { "input" }, * outputNames[] = { "output" };
    // 构造ONNXRuntime的OrtValue对象
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value txtTensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, txtToken.data(), txtToken.size(), txttokenShape.data(), txttokenShape.size());
    Ort::RunOptions run_options;
    vector<Ort::Value> Outputs = this->m_TxtEncoder->Run(run_options, inputNames, &txtTensor, 1, outputNames, 1);

    std::vector<int64_t>OutShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* txtembedding = (float*)Outputs[0].GetTensorMutableData<void>();

    embeddingshape.resize(OutShape.size());
    int totalsize = 1;
    for (int i = 0; i < OutShape.size(); i++)
    {
        totalsize *= OutShape[i];
        embeddingshape[i] = OutShape[i];
    }
    embedding.resize(totalsize);
    for (int i = 0; i < totalsize; i++)
    {
        embedding[i] = txtembedding[i];
    }
}


