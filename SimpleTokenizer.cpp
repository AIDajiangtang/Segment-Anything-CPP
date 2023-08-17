#include"SimpleTokenizer.h"

#include <fstream>
#include <sstream>


SimpleTokenizer::SimpleTokenizer()
{

}

SimpleTokenizer::~SimpleTokenizer()
{

}

/// <summary>
/// 初始化成员变量
/// </summary>
void SimpleTokenizer::Init()
{
    this->byte_encoder = this->BytesToUnicode();

    // 将 byte_encoder 转换为 byte_decoder
    for (const auto& kvp : this->byte_encoder) {
        this->byte_decoder[kvp.second] = kvp.first;
    }


    std::string bpePath = "";
    std::vector<std::tuple<std::string, std::string>> merges = LoadBPEMerges(bpePath);//加载BPE


      // 提取 byte_encoder 的值到 vocab
    std::vector<std::string> vocab;
    for (const auto& kvp : this->byte_encoder) {
        vocab.push_back(kvp.second);
        vocab.push_back(kvp.second + "</w>");
    }

    for (const auto& merge : merges) {
        vocab.push_back(std::get<0>(merge) + std::get<1>(merge));
    }

    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");


    for (int i = 0; i < vocab.size(); i++) {
        this->encoder[vocab[i]] = i;
    }

    for (const auto& kvp : encoder) {
        this->decoder[kvp.second] = kvp.first;
    }

    int index = 0;
    for (const auto& merge : merges) {
        this->bpe_ranks[merge] = index++;
    }

    this->cache["<|startoftext|>"] = "<|startoftext|>";

    this->pat = std::regex(R"(\s<\|startoftext\|\>|<\|endoftext\|\>|\s's|\s't|\s're|\s've|\s'm|\s'll|\s'd|[[:alpha:]]+|[[:digit:]]+|\S+)", std::regex_constants::icase);
}
std::vector<int64_t> SimpleTokenizer::Encode(const std::string& text) {
    std::vector<int64_t> bpeTokens;
    std::string cleanedText = whitespace_clean(text);
    std::transform(cleanedText.begin(), cleanedText.end(), cleanedText.begin(), [](char c) {
        return std::tolower(static_cast<unsigned char>(c));
        });

    std::regex pat(R"(<pattern>)");  // Replace <pattern> with your actual regex pattern

    std::sregex_iterator iter(cleanedText.begin(), cleanedText.end(), pat);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
        std::string token = "";
        for (char c : iter->str()) {
            token += byte_encoder[c];
        }

        std::stringstream ss(token);
        std::string bpeToken;
        while (std::getline(ss, bpeToken, ' ')) {
            bpeTokens.push_back(encoder[bpeToken]);
        }
    }

    return bpeTokens;
}
/// <summary>
/// 数字向量解码成文本
/// </summary>
std::string SimpleTokenizer::Decode(const std::vector<int>& tokens) {
    std::stringstream textBuilder;
    for (int token : tokens) {
        textBuilder << decoder[token];
    }
    std::string text = textBuilder.str();

    std::vector<uint8_t> byteList;
    for (char c : text) {
        byteList.push_back(static_cast<uint8_t>(byte_decoder[std::string(1, c)]));
    }
    std::string decodedText(byteList.begin(), byteList.end());
    std::replace(decodedText.begin(), decodedText.end(), '/', ' ');

    return decodedText;
}


std::string SimpleTokenizer::bpe(const std::string& token) {
    if (this->cache.count(token) > 0) {
        return cache[token];
    }

    std::vector<std::string> word;
    for (size_t i = 0; i < token.length() - 1; i++) {
        word.push_back(std::string(1, token[i]));
    }
    word.push_back(token.substr(token.length() - 1) + "</w>");

    std::vector<std::pair<std::string, std::string>> pairs = GetPairs(word);

    if (pairs.empty()) {
        return token + "</w>";
    }

    while (true) {
        auto minPair = std::min_element(pairs.begin(), pairs.end(),
            [&](const std::pair<std::string, std::string>& pair1, const std::pair<std::string, std::string>& pair2) {
                return bpe_ranks.count(pair1) ? bpe_ranks[pair1] : std::numeric_limits<double>::infinity() <
                    bpe_ranks.count(pair2) ? bpe_ranks[pair2] : std::numeric_limits<double>::infinity();
            });

        std::string first = minPair->first;
        std::string second = minPair->second;

        if (!bpe_ranks.count(std::make_pair(first, second))) {
            break;
        }

        std::vector<std::string> newWord;
        size_t i = 0;
        while (i < word.size()) {
            try {
                size_t j = std::find(word.begin() + i, word.end(), first) - word.begin();
                newWord.insert(newWord.end(), word.begin() + i, word.begin() + j);
                i = j;
            }
            catch (...) {
                newWord.insert(newWord.end(), word.begin() + i, word.end());
                break;
            }

            if (word[i] == first && i < word.size() - 1 && word[i + 1] == second) {
                newWord.push_back(first + second);
                i += 2;
            }
            else {
                newWord.push_back(word[i]);
                i += 1;
            }
        }

        word = newWord;

        if (word.size() == 1) {
            break;
        }
        else {
            pairs = GetPairs(newWord);
        }
    }

    std::string result = "";
    for (const std::string& w : word) {
        result += w + " ";
    }
    result = result.substr(0, result.size() - 1); // Remove trailing space

    cache[token] = result;
    return result;
}

std::map<int, std::string> SimpleTokenizer::BytesToUnicode() {
    std::vector<int> bs;
    std::vector<int> cs;

    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); b++) {
        bs.push_back(b);
        cs.push_back(b);
    }

    for (int b = static_cast<int>('¡'); b <= static_cast<int>('¬'); b++) {
        bs.push_back(b);
        cs.push_back(b);
    }

    for (int b = static_cast<int>('®'); b <= static_cast<int>('ÿ'); b++) {
        bs.push_back(b);
        cs.push_back(b);
    }

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    std::map<int, std::string> byteToUnicode;
    for (size_t i = 0; i < bs.size(); i++) {
        byteToUnicode[bs[i]] = std::string(1, static_cast<char>(cs[i]));
    }

    return byteToUnicode;
}
/// <summary>
/// 加载bpe文件
/// </summary>
std::vector<std::tuple<std::string, std::string>> SimpleTokenizer::LoadBPEMerges(const std::string& bpePath) {
    std::vector<std::tuple<std::string, std::string>> merges;

    std::ifstream fileStream(bpePath);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open file: " << bpePath << std::endl;
        return merges;
    }

    std::string line;
    while (std::getline(fileStream, line)) {
        std::istringstream streamReader(line);
        std::string merge[2];
        streamReader >> merge[0] >> merge[1];
        merges.emplace_back(merge[0], merge[1]);
    }

    fileStream.close();

    return merges;
}
std::vector<std::pair<std::string, std::string>> SimpleTokenizer::GetPairs(const std::vector<std::string>& words) {
    std::vector<std::pair<std::string, std::string>> pairs;

    for (std::size_t i = 0; i < words.size() - 1; ++i) {
        for (std::size_t j = i + 1; j < words.size(); ++j) {
            pairs.push_back(std::make_pair(words[i], words[j]));
        }
    }

    return pairs;
}
//std::unordered_set<std::pair<std::string, std::string>> SimpleTokenizer::GetPairs(const std::vector<std::string>& word) {
//    std::unordered_set<std::pair<std::string, std::string>> pairs;
//    std::string prevChar = word[0];
//    for (int i = 1; i < word.size(); i++) {
//        std::string currentChar = word[i];
//        pairs.insert(std::make_pair(prevChar, currentChar));
//        prevChar = currentChar;
//    }
//    return pairs;
//}
/// <summary>
/// 去掉空白格
/// </summary>
std::string SimpleTokenizer::whitespace_clean(std::string text) {
    // 使用正则表达式替换多个连续空白字符为单个空格
    text = std::regex_replace(text, std::regex("\\s+"), " ");

    // 去除字符串两端的空白字符
    text = std::regex_replace(text, std::regex("^\\s+|\\s+$"), "");

    return text;
}