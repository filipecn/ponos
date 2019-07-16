#version 440 core
layout(location = 0) out vec4 fragColor;

in vec3 fNormal;
in vec3 fPosition;

struct Light {
  vec3 position;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

struct Material {
  vec3 kAmbient;
  vec3 kDiffuse;
  vec3 kSpecular;
  float shininess;
};

layout(location = 2) uniform Light light;
layout(location = 6) uniform Material material;
layout(location = 10) uniform vec3 cameraPosition;

const float SEA_HEIGHT = 1.0;
const float PI = 3.141592;
const vec3 SEA_BASE = vec3(0.1, 0.19, 0.22);
const vec3 SEA_WATER_COLOR = vec3(0.8, 0.9, 0.6);

// lighting
float diffuse(vec3 n, vec3 l, float p) { return pow(dot(n, l) * 0.4 + 0.6, p); }
float specular(vec3 n, vec3 l, vec3 e, float s) {
  float nrm = (s + 8.0) / (PI * 8.0);
  return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

// sky
vec3 getSkyColor(vec3 e) {
  e.y = max(e.y, 0.0);
  return vec3(pow(1.0 - e.y, 2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4);
}

vec3 getSeaColor(vec3 p, vec3 n, vec3 l, vec3 eye, vec3 dist) {
  float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
  fresnel = pow(fresnel, 3.0) * 0.65;

  vec3 reflected = getSkyColor(reflect(eye, n));
  vec3 refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12;

  vec3 color = mix(refracted, reflected, fresnel);

  float atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);
  // color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten;

  color += vec3(specular(n, l, eye, 60.0));

  return color;
}

void main() {
  vec3 norm = -normalize(fNormal);
  vec3 lightDirection = normalize(light.position - fPosition);
  float diff = max(dot(norm, lightDirection), 0.0);
  vec3 viewDir = normalize(cameraPosition - fPosition);
  vec3 halfwayDirection = normalize(lightDirection + viewDir);
  // vec3 reflectDir = reflect(-lightDirection, norm);
  float spec =
      pow(max(dot(viewDir, halfwayDirection), 0.0), material.shininess);
  vec3 lightIntensity = (light.ambient * material.kAmbient +
                         light.diffuse * material.kDiffuse * diff +
                         light.specular * material.kSpecular * spec) *
                        vec3(1, 1, 1);
  fragColor = vec4(lightIntensity, 1.0);
  // fragColor = vec4(abs(fNormal.x), abs(fNormal.y), abs(fNormal.z), 1);
  // fragColor = vec4(getSeaColor(fPosition, fNormal, normalize(light.position),
  //                              viewDir, fPosition - cameraPosition),
  //                  1);
}