#ifndef AERGIA_IO_VULKAN_UTILS_H
#define AERGIA_IO_VULKAN_UTILS_H

#include <vector>

namespace aergia {
  namespace vulkan {

    inline VkResult createDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
      auto func = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
      if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
      } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
      }
    }

    inline void destroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
      auto func = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
      if (func != nullptr) {
        func(instance, callback, pAllocator);
      }
    }

    inline bool checkValidationLayerSupport(const std::vector<const char*>& validationLayers) {
      uint32_t layerCount;
      vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

      std::vector<VkLayerProperties> availableLayers(layerCount);
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

      for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
          if (strcmp(layerName, layerProperties.layerName) == 0) {
            layerFound = true;
            break;
          }
        }

        if (!layerFound) {
          return false;
        }
      }
      return true;
    }

    inline std::vector<const char*> getRequiredExtensions(bool enableValidationLayers) {
      std::vector<const char*> extensions;

      unsigned int glfwExtensionCount = 0;
      const char** glfwExtensions;
      glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

      for (unsigned int i = 0; i < glfwExtensionCount; i++) {
        extensions.push_back(glfwExtensions[i]);
      }

      if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
      }

      return extensions;
    }

  } // vulkan namespace
} // aergia namespace

#endif
