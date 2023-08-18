#include <vector>
#include <string>
#include <memory>
#include<iostream>
#include "opencv2/core.hpp"
#include "common.h"
#include <onnxruntime_cxx_api.h>
#include "SimpleTokenizer.h"
using namespace std;
using namespace cv;

class SAM_EXPORTS CLIP
{
public:
    CLIP();
    void ImgEncoder(cv::Mat img,std::vector<float>&embedding, std::vector<int64_t>& embeddingshape);
    void TxtEncoder(std::wstring txt, vector<int64_t>txttokenShape, std::vector<float>& embedding, std::vector<int64_t>& embeddingshape);
protected:
    void LoadONNXModel(std::wstring visualpath, std::wstring textualpath);

    std::unique_ptr<Ort::Session> m_ImgEncoder;//Image Encoder
    std::unique_ptr<Ort::Session> m_TxtEncoder;//Image Decoder
    std::unique_ptr<Ort::Env>m_env;
    std::unique_ptr<Ort::SessionOptions>m_sessionOption;

    std::unique_ptr<SimpleTokenizer> mTokenlizer;
};


