#include <aergia.h>

int main() {
  aergia::vulkan::VulkanApplication app(800, 800, "Hello Vulkan");
  app.enableValidationLayers = false;
  app.run();
  return 0;
}
