#include "camera.h"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

void Camera::update()
{
    glm::mat4 cameraRotation = GetHorizontalRotationMatrix();
    const float movementSpeed = 0.02f;
    glm::vec3 xzVelocity = velocity; xzVelocity.y = 0.0f;
    glm::vec3 yVelocity = velocity; yVelocity.x = 0.0f; yVelocity.z = 0.0f;
    
    position += glm::vec3(cameraRotation * glm::vec4(xzVelocity * movementSpeed, 0.0f));
    position += yVelocity * movementSpeed, 0.0f;
}

void Camera::processSDLEvent(SDL_Event &e)
{
    if (e.type == SDL_KEYDOWN)
    {
        if (e.key.keysym.sym == SDLK_w) { velocity.z = -1; }
        if (e.key.keysym.sym == SDLK_s) { velocity.z = 1; }
        if (e.key.keysym.sym == SDLK_a) { velocity.x = -1; }
        if (e.key.keysym.sym == SDLK_d) { velocity.x = 1; }
        if (e.key.keysym.sym == SDLK_SPACE) { velocity.y = 1; }
        if (e.key.keysym.sym == SDLK_LSHIFT) { velocity.y = -1; }
    }

    if (e.type == SDL_KEYUP)
    {
        if (e.key.keysym.sym == SDLK_w) { velocity.z = 0; }
        if (e.key.keysym.sym == SDLK_s) { velocity.z = 0; }
        if (e.key.keysym.sym == SDLK_a) { velocity.x = 0; }
        if (e.key.keysym.sym == SDLK_d) { velocity.x = 0; }
        if (e.key.keysym.sym == SDLK_SPACE) { velocity.y = 0; }
        if (e.key.keysym.sym == SDLK_LSHIFT) { velocity.y = 0; }
    }

    if (e.type == SDL_MOUSEMOTION)
    {
        const float sens = 0.1f;

        yaw += (float)e.motion.xrel * sens;
        if (yaw >= 360.0f) yaw -= 360.0f;
        if (yaw < 0.0f) yaw += 360.0f;
        pitch -= (float)e.motion.yrel * sens;
        if (pitch > 89.0f) pitch = 89.9f;
        if (pitch < -89.0f) pitch = -89.0f;
    }
}

glm::mat4 Camera::getViewMatrix()
{
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix()
{
    glm::quat pitchRotation = glm::angleAxis(glm::radians(pitch), glm::vec3 { 1.0f, 0.0f, 0.0f });
    glm::quat yawRotation = glm::angleAxis(glm::radians(yaw), glm::vec3 { 0.0f, -1.0f, 0.0f });
    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

glm::mat4 Camera::GetHorizontalRotationMatrix()
{
    glm::quat yawRotation = glm::angleAxis(glm::radians(yaw), glm::vec3 { 0.0f, -1.0f, 0.0f });
    return glm::toMat4(yawRotation);
}

glm::vec3 Camera::GetFront()
{
    glm::mat3 rotation = glm::mat3(getRotationMatrix());
    return -rotation[2];
}
