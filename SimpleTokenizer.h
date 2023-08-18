#pragma once
#include <iostream>
#include <tuple>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
namespace std
{
    struct WstringHash {
        std::size_t operator()(const std::wstring& str) const {
            return std::hash<std::wstring>()(str);
        }
    };
    struct TupleHash {
        template <typename T>
        std::size_t operator()(const T& tuple) const {
            std::size_t seed = 0;
            hash_combine(seed, std::get<0>(tuple));
            hash_combine(seed, std::get<1>(tuple));
            return seed;
        }

        template <typename T>
        void hash_combine(std::size_t& seed, const T& value) const {
            seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    };
}

/// <summary>
/// 文本转换为数字向量
/// </summary>
class SAM_EXPORTS SimpleTokenizer
{
public:
    SimpleTokenizer();
    virtual ~SimpleTokenizer();

    void Init();//初始化成员变量

    std::vector<int64_t> tokenlize(std::wstring textpromot);
protected:
    std::wregex pat;//正则表达式
    const int contextLength = 77;//上下文长度，句子长度
    std::unordered_map<std::wstring, int> encoder;
    std::unordered_map<int, std::wstring> decoder;

    std::unordered_map<int, std::wstring> byte_encoder;
    std::unordered_map<std::wstring, int> byte_decoder;

    std::unordered_map<std::wstring, std::wstring> cache;
    std::unordered_map<std::tuple<std::wstring, std::wstring>, int, std::TupleHash> bpe_ranks;

    std::wstring bpe(const std::wstring& token);
    std::wstring whitespace_clean(std::wstring text);
    std::unordered_map<int, std::wstring> BytesToUnicode();
    std::wstring Decode(const std::vector<int>& tokens);
    std::vector<int64_t> Encode(const std::wstring& text);
    std::vector<std::tuple<std::wstring, std::wstring>> LoadBPEMerges(const std::wstring& bpePath);
    std::vector<std::pair<std::wstring, std::wstring>> GetPairs(const std::vector<std::wstring>& words);
    std::vector<std::wstring> split(const std::wstring& str, wchar_t delimiter);
    std::tuple<std::wstring, std::wstring> FindFirstPair(const std::vector<std::pair<std::wstring, std::wstring>>& pairs, const std::unordered_map<std::tuple<std::wstring, std::wstring>, int, std::TupleHash>& bpe_ranks);
};