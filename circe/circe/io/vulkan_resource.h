#ifndef AERGIA_IO_VULKAN_RESOURCE_H
#define AERGIA_IO_VULKAN_RESOURCE_H

// code based on vulkan-tutorial.com
#include <functional>

namespace aergia {
  namespace vulkan {

    template <typename T>
    class VulkanResource {
    public:
      VulkanResource()
      : VulkanResource([](T, VkAllocationCallbacks*){}) {}
      VulkanResource(std::function<void(T, VkAllocationCallbacks*)> df) {
        this->deleter = [=](T obj) {
          df(obj, nullptr);
        };
      }
      VulkanResource(const VulkanResource<VkInstance>& instance,
      std::function<void(VkInstance, T, VkAllocationCallbacks*)> df) {
        this->deleter = [&instance, df](T obj) {
          df(instance, obj, nullptr);
        };
      }
      VulkanResource(const VulkanResource<VkDevice>& device,
      std::function<void(VkDevice, T, VkAllocationCallbacks*)> df) {
        this->deleter = [&device, df](T obj) {
          df(device, obj, nullptr);
        };
      }
      ~VulkanResource() {
        cleanup();
      }
      const T* operator&() const {
        return &object;
      }
      T* replace() {
        cleanup();
        return &object;
      }
      operator T() const {
        return object;
      }
      void operator=(T rhs) {
        if (rhs != object) {
          cleanup();
          object = rhs;
        }
      }
      template<typename V>
      bool operator==(V rhs) {
        return object == T(rhs);
      }
    private:
      void cleanup() {
        if (object != VK_NULL_HANDLE) {
          deleter(object);
        }
        object = VK_NULL_HANDLE;
      }

      T object{VK_NULL_HANDLE};
      std::function<void(T)> deleter;
    };

  } // vulkan namespace
} // aergia namespace

#endif
