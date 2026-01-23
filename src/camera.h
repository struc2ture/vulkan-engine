#pragma once

#include <SDL_events.h>
#include <vk_types.h>

class Camera {
public:
    glm::vec3 velocity;
    glm::vec3 position;

    float pitch { 0.0f };
    float yaw { 0.0f };

    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();
    glm::mat4 GetHorizontalRotationMatrix();

    glm::vec3 GetFront();
    glm::vec3 GetUp();
    glm::vec3 GetRight();

    void processSDLEvent(SDL_Event& e);
    void processSDLEventConsoleMode(SDL_Event& e);
    
    void update();

private:
    bool _orbitingMotion { false };
    bool _panningMotion { false };
    
    float _accumDeltaX { 0.0f };
    float _accumDeltaY { 0.0f };

    glm::vec3 _orbitCenter{ 0.0f, 0.0f, 0.0f};
    float _orbitDist{ 5.0f };
};
