#ifndef AERGIA_IO_VULKAN_APPLICATION_H
#define AERGIA_IO_VULKAN_APPLICATION_H

#include "utils/open_gl.h"
#include "io/vulkan_resource.h"
#include "io/vulkan_utils.h"

#include <string>
#include <vector>

namespace aergia {
  namespace vulkan {

    class VulkanApplication {
    public:
      VulkanApplication(int w, int h, const char* windowTitle)
      : initialized(false), width(w), height(h) {
        title = std::string(windowTitle);
      }

      bool init();
      void run();

      bool enableValidationLayers;
    private:
      static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
      VkDebugReportFlagsEXT flags,
      VkDebugReportObjectTypeEXT objType,
      uint64_t obj,
      size_t location,
      int32_t code,
      const char* layerPrefix,
      const char* msg,
      void* userData) {

        std::cerr << "validation layer: " << msg << std::endl;

        return VK_FALSE;
      }

      struct QueueFamilyIndices {
        int graphicsFamily = -1;

        bool isComplete() {
          return graphicsFamily >= 0;
        }
      };

      bool initVulkan();
      void createInstance();
      void setupDebugCallback();
      void pickPhysicalDevice();
      bool isDeviceSuitable(VkPhysicalDevice device);
      QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

      bool initialized;
      int width, height;
      std::string title;
      GLFWwindow* window;
      const std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation"
      };
      VulkanResource<VkInstance> instance {vkDestroyInstance};
      VulkanResource<VkDebugReportCallbackEXT> callback{instance, destroyDebugReportCallbackEXT};
      VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    };

  } // vulkan namespace
} // aergia namespace

#endif
