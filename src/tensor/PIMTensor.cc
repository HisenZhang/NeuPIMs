#include "PIMTensor.h"

#include "../allocator/AddressAllocator.h"

PIMTensor::PIMTensor(std::string name, uint32_t ch, std::vector<uint32_t> dims,
                     PIMTensorKVType kv_type, bool produced) {
    _name = name;
    _ch = ch;
    _dims = dims;  // [h, seq_len, d_k] or [h, d_k, seq_len]
    _precision = Config::global_config.precision;
    _produced = produced;
    _kv_type = kv_type;

    auto alloc = KVCacheAlloc::GetInstance();
    _seq_len = kv_type == PIMTensorKVType::KEY ? dims[2] : dims[1];
    _bank_per_ch = alloc->_bank_per_ch;
    _num_ele_per_row = alloc->_num_ele_per_row;
    _E = Config::global_config.model_n_embd;

    uint32_t num_alloc_iter = 0;  // calculate # of allocation iterations based on seq_len.
    if (kv_type == PIMTensorKVType::KEY) {
        // KEY: allocate (E / C) rows
        _num_rows_per_alloc = ceil((double)_E / (double)_num_ele_per_row);
        num_alloc_iter = ceil((double)_seq_len / (double)_bank_per_ch);
    } else {
        // VALUE: allocate (E / bank_per_ch) rows
        _num_rows_per_alloc = ceil((double)_E / (double)_bank_per_ch);
        num_alloc_iter = ceil((double)_seq_len / (double)_num_ele_per_row);
    }

    uint32_t num_required_alloc = num_alloc_iter * _num_rows_per_alloc;

    for (int i = 0; i < num_required_alloc; ++i) _rows.push_back(alloc->allocate(ch));
}

// addr_type PIMTensor::get_addr(std::vector<uint32_t> indexes) { return 0; }

addr_type PIMTensor::get_addr(std::vector<uint32_t> indexes) {
    if (indexes.size() != 3) return 0;

    uint32_t seq_idx = indexes[1];
    uint32_t embd_idx = indexes[2];
    
    constexpr uint32_t bank_offset = 12;
    constexpr uint32_t row_offset = 20;

    if (_kv_type == PIMTensorKVType::KEY) {
        // KEY: spread E dimension across banks
        uint32_t bank_idx = embd_idx / _num_ele_per_row;
        uint32_t bank_offset_val = embd_idx % _num_ele_per_row;
        uint32_t row_idx = seq_idx * _num_rows_per_alloc + (bank_idx / _bank_per_ch);
        
        // Get base address for this entry
        if (row_idx >= _rows.size()) return 0;
        addr_type base_addr = _rows[row_idx];
        
        // Modify bank bits while preserving row address
        base_addr &= ~(((1 << 2) - 1) << bank_offset); // Clear bank bits
        base_addr |= ((bank_idx % _bank_per_ch) << bank_offset); // Set new bank
        
        return base_addr | (bank_offset_val * _precision);
    } else {
        // VALUE: spread sequence length across banks
        uint32_t bank_idx = seq_idx / _num_ele_per_row;
        uint32_t bank_offset_val = seq_idx % _num_ele_per_row;
        uint32_t row_idx = (embd_idx % (_E / _bank_per_ch)) * _num_rows_per_alloc + bank_idx;
        
        if (row_idx >= _rows.size()) return 0;
        addr_type base_addr = _rows[row_idx];
        
        // Modify bank bits for value layout
        base_addr &= ~(((1 << 2) - 1) << bank_offset);
        base_addr |= ((embd_idx / (_E / _bank_per_ch)) << bank_offset);
        
        return base_addr | (bank_offset_val * _precision);
    }
}

std::vector<addr_type> PIMTensor::get_all_addrs() {
    std::vector<addr_type> ret;
    return ret;
}

uint32_t PIMTensor::get_allocated_seq_len() {
    if (_kv_type == PIMTensorKVType::KEY)
        return ceil((double)_seq_len / (double)_bank_per_ch) * _bank_per_ch;
    else
        return ceil((double)_seq_len / (double)_num_ele_per_row) * _num_ele_per_row;
}

void PIMTensor::add_token() {
    _seq_len++;
    if (_kv_type == PIMTensorKVType::KEY)
        _dims[2]++;
    else
        _dims[1]++;

    if (_seq_len <= get_allocated_seq_len()) return;

    for (int i = 0; i < _num_rows_per_alloc; ++i)
        _rows.push_back(KVCacheAlloc::GetInstance()->allocate(_ch));
}

uint32_t PIMTensor::get_num_rows() { return _rows.size(); }

uint32_t PIMTensor::get_channel() { return _ch; }

std::vector<uint64_t> PIMTensor::get_rows() { return _rows; }