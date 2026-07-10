#include "vectordb/collection.hpp"

#include <stdexcept>

namespace vectordb
{

    void Collection::save(const std::filesystem::path &path) const
    {
        (void)path;
        throw std::logic_error("TODO: implement Collection::save");
    }

    std::unique_ptr<Collection> Collection::load(const std::filesystem::path &path)
    {
        (void)path;
        throw std::logic_error("TODO: implement Collection::load");
    }

}
