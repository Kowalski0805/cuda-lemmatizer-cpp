//
// Created by Illya on 07.04.2025.
//

#include "../../include/icu_lowercase.h"
#include <unicode/unistr.h>
#include <unicode/locid.h>
#include <string>

std::string lowercase_ukr(const std::string& input) {
    icu::UnicodeString ustr(input.c_str(), "UTF-8");
    ustr.toLower(icu::Locale("uk"));
    std::string result;
    ustr.toUTF8String(result);
    return result;
}

// extern "C" {
//     const char* lowercase_ukr(const char* input) {
//         static std::string result;
//         result = lowercase_ukr(std::string(input));  // calls your real one
//         return result.c_str();
//     }
// }