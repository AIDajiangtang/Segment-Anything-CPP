#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include "opencv2/core.hpp"
#include "common.h"
#include "onnxruntime_cxx_api.h"
using namespace std;
using namespace cv;

/// <summary>
/// Image Transform
/// Normalization and Resize
/// </summary>
class SAM_EXPORTS Transform
{
public:
    Transform(int targetwidth);
    cv::Mat TransformImage(cv::Mat& orgimg);

protected:
    /// <summary>
    /// Get Transformed Image Size
    /// </summary>
    void GetPreprocessShape(int oldw, int oldh, int long_side_length, int& neww, int& newh);
    int m_targetWidth;
};
/// <summary>
/// Semgement Anything C++ Inference
/// </summary>
class SAM_EXPORTS SAM
{
public:
    SAM(int targetsize);
    /// <summary>
    /// Image Encoding
    /// </summary>
    void ImageEncode(string imgpath);  
    void Decoder(std::vector<float>promotions,std::vector<float>labels,int promotionCount);
  
protected:
    std::unique_ptr<Ort::Session> m_Encoder;//Image Encoder
    std::unique_ptr<Ort::Session> m_Decoder;//Image Decoder
    std::unique_ptr<Transform> m_Transform;
    std::unique_ptr<Ort::Env>m_env;
    std::unique_ptr<Ort::SessionOptions>m_sessionOption;
    std::unique_ptr<float> m_ImgEmbedding;
    std::vector<int64_t> m_ImgEmbeddingshape;
    std::unique_ptr<float>m_resultMask;
    int m_orgWid;
    int m_orgHei;
    int m_targetSize;
    /// <summary>
    /// Load ONNX PreTrained Models
    /// </summary>
    void LoadOnnxModel();
};


