#include <vector>
#include <string>
#include <memory>
#include<iostream>
#include "opencv2/core.hpp"
#include "common.h"
#include <onnxruntime_cxx_api.h>
using namespace std;
using namespace cv;

class SAM_EXPORTS CLIP
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
