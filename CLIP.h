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

class CLIP
{
public:
    CLIP();
    void ImgEncoder(cv::Mat img,std::vector<float>&embedding, std::vector<int64_t>& embeddingshape);
    void TxtEncoder(std::vector<int64_t>txttoken, vector<int64_t>txttokenShape, std::vector<float>& embedding, std::vector<int64_t>& embeddingshape);
protected:
    void LoadONNXModel(std::wstring visualpath, std::wstring textualpath);

    std::unique_ptr<Ort::Session> m_ImgEncoder;//Image Encoder
    std::unique_ptr<Ort::Session> m_TxtEncoder;//Image Decoder
    std::unique_ptr<Ort::Env>m_env;
    std::unique_ptr<Ort::SessionOptions>m_sessionOption;
};

class Tokenlizer
{
public:
    Tokenlizer()
    {
        //Load Vocab
    }
    std::vector<int64_t>Tokenlize(std::vector<string>txts)
    {
        //"a diagram", "a dog", "a cat"
        std::vector<int64_t> txtToken = {
          49406,320,22697,49407,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,
          49406,320,1929,49407,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,
          49406,320,2368,49407,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0 };

        return txtToken;
    }
};
