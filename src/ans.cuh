
#pragma once

int compress_stream_gpu(const std::vector<int32_t>& data, const char* name);

int compress_stream_gpu(const std::vector<int8_t>& data, const char* name);

void normalize_freqs(std::vector<uint32_t>& freqs, std::vector<uint32_t>& starts);