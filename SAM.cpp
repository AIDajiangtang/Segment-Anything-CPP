#include"SAM.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

/// <summary>
/// Image Transform
/// Normalization and Resize
/// </summary>

Transform::Transform(int targetwidth)
{
      //Resize org image H*W to targetwidth*targetwidth,zero padding
      this->m_targetWidth = targetwidth;
 }
cv::Mat Transform::TransformImage(cv::Mat&orgimg)
{
    int orgwidth = orgimg.cols;    // 小图像宽度
    int orgheight = orgimg.rows;   // 小图像高度

    int neww = 0;
    int newh = 0;
    this->GetPreprocessShape(orgwidth, orgheight, this->m_targetWidth, neww, newh);

    //Resize
    cv::Mat resizeImage;
    cv::resize(orgimg, resizeImage, cv::Size(neww, newh), cv::INTER_AREA);

    int pad_h = this->m_targetWidth - resizeImage.rows;
    int pad_w = this->m_targetWidth - resizeImage.cols;

    //padding to 1024*1024
    cv::Mat paddingImage;
    cv::copyMakeBorder(resizeImage, paddingImage, 0, pad_h, 0, pad_w, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    //Normalization
    paddingImage.convertTo(paddingImage, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(paddingImage, mean, stddev);

    for (int i = 0; i < paddingImage.rows; i++) {
        for (int j = 0; j < paddingImage.cols; j++) {
            float b = paddingImage.at<cv::Vec3f>(i, j)[0];
            float g = paddingImage.at<cv::Vec3f>(i, j)[1];
            float r = paddingImage.at<cv::Vec3f>(i, j)[2];
            paddingImage.at<cv::Vec3f>(i, j)[0] = (b - mean[0]) / stddev[0];
            paddingImage.at<cv::Vec3f>(i, j)[1] = (g - mean[1]) / stddev[1];
            paddingImage.at<cv::Vec3f>(i, j)[2] = (r - mean[2]) / stddev[2];
        }
    }
  
    return paddingImage;
}



/// <summary>
/// Get Transformed Image Size
/// </summary>
void Transform::GetPreprocessShape(int oldw, int oldh, int long_side_length,  int& neww,  int& newh)
{      
    float scale = long_side_length * 1.0f / std::max(oldh, oldw);
    float newht = oldh * scale;
    float newwt = oldw * scale;

    neww = (int)(newwt + 0.5);
    newh = (int)(newht + 0.5);
}


SAM::SAM(int targetsize) {
    this->LoadOnnxModel();
    this->m_targetSize = targetsize;
    this->m_Transform = std::unique_ptr<Transform>(new Transform(this->m_targetSize));
}
/// <summary>
/// Image Encoding
/// </summary>
void SAM::ImageEncode(string imgpath)
{
    cv::Mat img = cv::imread(imgpath);
    this->m_orgWid = img.cols;
    this->m_orgHei = img.rows;

    cv::Mat transformed = this->m_Transform->TransformImage(img);

    std::vector<float>imgv; imgv.resize(this->m_targetSize * this->m_targetSize *3);
    for (int i = 0;i < transformed.cols; i++)
    {
        for (int j=0;j< transformed.rows;j++)
        {
            int index = j * this->m_targetSize + i;
            imgv[index] = transformed.at<cv::Vec3f>(j,i)[0];
            imgv[this->m_targetSize * this->m_targetSize + index] = transformed.at<cv::Vec3f>(j, i)[1];
            imgv[2 * this->m_targetSize * this->m_targetSize + index] = transformed.at<cv::Vec3f>(j, i)[2];
        }
    }
    vector<int64_t> inputShape{ 1, 3, this->m_targetSize, this->m_targetSize };
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 构造ONNXRuntime的OrtValue对象
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgv.data(), imgv.size(), inputShape.data(), inputShape.size());
    const char* inputNamesPre[] = { "x" }, * outputNamesPre[] = { "image_embeddings" };
    Ort::RunOptions run_options;
    vector<Ort::Value> outputs = this->m_Encoder->Run(run_options, inputNamesPre, &inputTensor, 1, outputNamesPre, 1);

    this->m_ImgEmbeddingshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* embedding = (float*)outputs[0].GetTensorMutableData<void>();

    int64_t totalsize = 1;
    for (int i=0;i< this->m_ImgEmbeddingshape.size();i++)
    {
        totalsize *= this->m_ImgEmbeddingshape[i];
    }
    this->m_ImgEmbedding = std::unique_ptr<float>(new float[totalsize]);
    std::memcpy(this->m_ImgEmbedding.get(), embedding, totalsize*sizeof(float));

}
void SAM::Decoder(std::vector<float>promotions,std::vector<float>labels,int promotionCount)
{
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    int64_t totalsize = 1;
    for (int i = 0; i < this->m_ImgEmbeddingshape.size(); i++)
    {
        totalsize *= this->m_ImgEmbeddingshape[i];
    }
    std::vector<Ort::Value> inputTensor;
    //image_embeddings input tensor
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, this->m_ImgEmbedding.get(), totalsize, this->m_ImgEmbeddingshape.data(), this->m_ImgEmbeddingshape.size()));
    //point_coords input tensor
    vector<int64_t> pointShape{ 1, promotionCount, 2 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, promotions.data(), promotions.size(), pointShape.data(), pointShape.size()));
    //point_labels input tensor
    vector<int64_t> pointLabelShape{ 1, promotionCount };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, labels.data(), labels.size(), pointLabelShape.data(), pointLabelShape.size()));
    //mask_input input tensor 
    std::vector<float> mask(256 * 256, 0.0f);
    vector<int64_t>maskShape{ 1, 1, 256, 256 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, mask.data(), mask.size(), maskShape.data(), maskShape.size()));
    //has_mask_input input tensor
    std::vector<float> hasMask(1, 0.0f);
    vector<int64_t>hasMaskShape{ 1 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMask.data(), hasMask.size(), hasMaskShape.data(), hasMaskShape.size()));
    //orig_im_size input tensor
    std::vector<float> origiImSize{ (float)this->m_orgWid, (float)this->m_orgHei };
    vector<int64_t>origImSizeShape{ 2 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, origiImSize.data(), origiImSize.size(), origImSizeShape.data(), origImSizeShape.size()));

    const char* inputNamesPre[] = { "image_embeddings","point_coords","point_labels","mask_input","has_mask_input","orig_im_size" };
    const char* outputNamesPre[] = { "masks","iou_predictions","low_res_masks" };
    Ort::RunOptions run_options;
    vector<Ort::Value> outputs = this->m_Decoder->Run(run_options, inputNamesPre, inputTensor.data(), inputTensor.size(), outputNamesPre, 3);

    vector<int64_t>mask0Shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* resultmask = (float*)outputs[0].GetTensorMutableData<void>();
    vector<int64_t>mask1Shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    vector<int64_t>mask2Shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();

    totalsize = 1;
    for (int i = 0; i < mask0Shape.size(); i++)
    {
        totalsize *= mask0Shape[i];
    }
    this->m_resultMask = std::unique_ptr<float>(new float[totalsize]);
    std::memcpy(this->m_resultMask.get(), resultmask, totalsize*sizeof(float));

}

/// <summary>
/// Load ONNX PreTrained Models
/// </summary>
void SAM::LoadOnnxModel()
{
    this->m_env = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Sam"));
    this->m_sessionOption = std::unique_ptr<Ort::SessionOptions>(new Ort::SessionOptions());
    this->m_sessionOption->SetIntraOpNumThreads(1);
    this->m_sessionOption->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring encoderwstr = L"D:\\SAM\\encoder-quant.onnx";
    this->m_Encoder = std::unique_ptr<Ort::Session>(new  Ort::Session(*this->m_env.get(), encoderwstr.c_str(), *this->m_sessionOption.get()));

    std::wstring decoderwstr = L"D:\\SAM\\decoder-quant.onnx";
    this->m_Decoder = std::unique_ptr<Ort::Session>(new  Ort::Session(*this->m_env.get(), decoderwstr.c_str(), *this->m_sessionOption.get()));
}
