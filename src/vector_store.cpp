#include "vectordb/vector_store.hpp"

#include <stdexcept>

namespace vectordb
{

    VectorStore::VectorStore(std::size_t dim)
        : dim_(dim)
    {
        if (dim_ == 0)
        {
            throw std::invalid_argument("Vector dimension must be greater than zero");
        }
    }

    std::uint64_t VectorStore::add(std::span<const float> vector)
    {
        if (vector.size() != dim_)
        {
            throw std::invalid_argument("Vector dimension does not match store dimension");
        }

        const auto internal_id = static_cast<std::uint64_t>(size());
        data_.insert(data_.end(), vector.begin(), vector.end());
        return internal_id;
    }

    const float *VectorStore::get(std::uint64_t internal_id) const
    {
        if (internal_id >= size())
        {
            throw std::out_of_range("Vector id is out of range");
        }

        return data_.data() + (internal_id * dim_);
    }

    std::size_t VectorStore::size() const
    {
        return data_.size() / dim_;
    }

    std::size_t VectorStore::dim() const
    {
        return dim_;
    }

}
