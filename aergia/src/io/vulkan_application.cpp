#include "io/vulkan_application.h"

namespace aergia {
  namespace vulkan {

    bool VulkanApplication::init() {
      if (!glfwInit())
      return false;
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
      window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
      if (!window){
        glfwTerminate();
        return false;
      }
      initVulkan();
      return true;
    }

    bool VulkanApplication::initVulkan() {
      createInstance();
      setupDebugCallback();
      pickPhysicalDevice();
      initialized = true;
      return true;
    }

    void VulkanApplication::createInstance() {
      if (enableValidationLayers && !checkValidationLayerSupport(validationLayers)) {
        throw std::runtime_error("validation layers requested, but not available!");
      }
      VkApplicationInfo appInfo = {};
      appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      appInfo.pApplicationName = title.c_str();
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 3);
      appInfo.pEngineName = "No Engine";
      appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 3);
      appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 3);

      VkInstanceCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      createInfo.pApplicationInfo = &appInfo;

      //unsigned int glfwExtensionCount = 0;
      //const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

      if (enableValidationLayers) {
        createInfo.enabledLayerCount = validationLayers.size();
        createInfo.ppEnabledLayerNames = validationLayers.data();
      } else {
        createInfo.enabledLayerCount = 0;
      }

      auto extensions = getRequiredExtensions(enableValidationLayers);
      createInfo.enabledExtensionCount = extensions.size();
      createInfo.ppEnabledExtensionNames = extensions.data();

      if (vkCreateInstance(&createInfo, nullptr, instance.replace()) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
      }
    }

    void VulkanApplication::setupDebugCallback() {
      if (!enableValidationLayers) return;
      VkDebugReportCallbackCreateInfoEXT createInfo = {};
      //createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
      createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
      createInfo.pfnCallback = debugCallback;
      if (createDebugReportCallbackEXT(instance, &createInfo, nullptr, callback.replace()) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug callback!");
      }
    }

    void VulkanApplication::pickPhysicalDevice() {
      uint32_t deviceCount = 0;
      vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
      if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
      }
      std::vector<VkPhysicalDevice> devices(deviceCount);
      vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
      for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
          physicalDevice = device;
          break;
        }
      }

      if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
      }
    }

    bool VulkanApplication::isDeviceSuitable(VkPhysicalDevice device) {
			QueueFamilyIndices indices = findQueueFamilies(device);
			return indices.isComplete();
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
			VkPhysicalDeviceFeatures deviceFeatures;
			vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
			return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
				deviceFeatures.geometryShader;
		}

    VulkanApplication::QueueFamilyIndices VulkanApplication::findQueueFamilies(VkPhysicalDevice device) {
			QueueFamilyIndices indices;
			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
			int i = 0;
			for (const auto& queueFamily : queueFamilies) {
				if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
					indices.graphicsFamily = i;
				}

				if (indices.isComplete()) {
					break;
				}

				i++;
			}
			return indices;
		}

    void VulkanApplication::run() {
      if(!initialized)
      init();
      while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
      }
    }
  } // vulkan namespace
} // aergia namespace
