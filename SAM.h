#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/opencv_modules.hpp"
#include <iostream>
#include <fstream> //for file operations
#include <vector>
#include <memory>
#include "opencv2/core/ocl.hpp"
#include <opencv2/core/utils/logger.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include "opencv2/calib3d.hpp"
#include <onnxruntime_cxx_api.h>
using namespace cv::detail;
using namespace std;
using namespace cv;


/// <summary>
/// Semgement Anything C++ Inference
/// </summary>
class SAM
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

/// <summary>
/// Image Transform
/// Normalization and Resize
/// </summary>
class Transform
{
public:
    Transform(int targetwidth);
    cv::Mat TransformImage(cv::Mat&orgimg);
  
protected:
    /// <summary>
    /// Get Transformed Image Size
    /// </summary>
    void GetPreprocessShape(int oldw, int oldh, int long_side_length,  int& neww,  int& newh);
    int m_targetWidth;
};
