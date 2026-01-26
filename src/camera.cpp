#include "camera.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "vk_types.h"

void Camera::update()
{
    if (_panningMotion && _orbitingMotion)
    {
        _orbitCenter -= _accumDeltaX * 0.08f * GetRight();
        _orbitCenter -= _accumDeltaY * 0.08f * GetUp();

        float yawRad   = glm::radians(yaw);
        float pitchRad = glm::radians(pitch);

        glm::vec3 translation{};
        translation.x = _orbitDist * cos(-pitchRad) * sin(-yawRad);
        translation.y = _orbitDist * sin(-pitchRad);
        translation.z = _orbitDist * cos(-pitchRad) * cos(-yawRad);

        position = _orbitCenter + translation;

        _accumDeltaX = 0.0f;
        
        _accumDeltaY = 0.0f;
    }
    else if (_dollyMotion && _orbitingMotion)
    {
        _orbitDist -= _accumDeltaY * 0.15f;

        float yawRad   = glm::radians(yaw);
        float pitchRad = glm::radians(pitch);

        glm::vec3 translation{};
        translation.x = _orbitDist * cos(-pitchRad) * sin(-yawRad);
        translation.y = _orbitDist * sin(-pitchRad);
        translation.z = _orbitDist * cos(-pitchRad) * cos(-yawRad);

        position = _orbitCenter + translation;

        _accumDeltaX = 0.0f;
        _accumDeltaY = 0.0f;
    }
    else if (_orbitingMotion)
    {
        //fmt::println("Orbit: {} {}", _accumDeltaX, _accumDeltaY);

        //glm::mat4 rot {1.0f};

        //rot = glm::rotate(rot, -_accumDeltaY * 0.1f, glm::vec3{1.0f, 0.0f, 0.0f});
        //rot = glm::rotate(rot, -_accumDeltaX * 0.1f, glm::vec3{0.0f, 1.0f, 0.0f});

        //glm::vec4 p { position, 1.0f};

        //p = rot * p;

        //position = glm::vec3(p);

        //glm::mat3 lookAt = glm::lookAt(position, glm::vec3 {0.0f, 0.0f, 0.0f}, glm::vec3 {0.0f, 1.0f, 0.0f});

        //glm::vec3 forward = glm::normalize(glm::vec3(0.0f) - position);
        //pitch = glm::degrees(std::asin(forward.y));
        //yaw   = glm::degrees(std::atan2(forward.x, -forward.z));
        //fmt::println("Yaw: {}; Pitch; {}", yaw, pitch);

        //yaw -= glm::degrees(_accumDeltaX * 0.1f);
        //pitch += glm::degrees(_accumDeltaY * 0.1f);

        yaw   += _accumDeltaX * 2.0f;
        pitch += _accumDeltaY * 2.0f;

        pitch = glm::clamp(pitch, -89.0f, 89.0f);

        float yawRad   = glm::radians(yaw);
        float pitchRad = glm::radians(pitch);

        glm::vec3 translation{};
        translation.x = _orbitDist * cos(-pitchRad) * sin(-yawRad);
        translation.y = _orbitDist * sin(-pitchRad);
        translation.z = _orbitDist * cos(-pitchRad) * cos(-yawRad);

        position = _orbitCenter + translation;

        _accumDeltaX = 0.0f;
        _accumDeltaY = 0.0f;
    }
    else
    {
        glm::mat4 cameraRotation = GetHorizontalRotationMatrix();
        const float movementSpeed = 0.02f;
        glm::vec3 xzVelocity = velocity; xzVelocity.y = 0.0f;
        glm::vec3 yVelocity = velocity; yVelocity.x = 0.0f; yVelocity.z = 0.0f;
    
        position += glm::vec3(cameraRotation * glm::vec4(xzVelocity * movementSpeed, 0.0f));
        position += yVelocity * movementSpeed, 0.0f;
    }
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

void Camera::processSDLEventConsoleMode(SDL_Event &e)
{
    if (e.type == SDL_MOUSEBUTTONDOWN)
    {
        if (e.button.button == 2)
        {
            _orbitingMotion = true;
            _orbitCenter = position + GetFront() * _orbitDist;
            _accumDeltaX = 0.0f;
            _accumDeltaY = 0.0f;
            //SDL_SetRelativeMouseMode((SDL_bool)true);
            //SDL_GetMouseState(&_prevMouseX, &_prevMouseY);
        }
    }
    else if (e.type == SDL_MOUSEBUTTONUP)
    {
        if (e.button.button == 2)
        {
            _orbitingMotion = false;
            //SDL_SetRelativeMouseMode((SDL_bool)false);
        }
    }

    if (e.type == SDL_KEYDOWN)
    {
        if (e.key.keysym.sym == SDLK_LSHIFT)
        {
            _panningMotion = true;
            _accumDeltaX = 0.0f;
            _accumDeltaY = 0.0f;
        }
    }
    else if (e.type == SDL_KEYUP)
    {
        if (e.key.keysym.sym == SDLK_LSHIFT)
        {
            _panningMotion = false;
        }
    }

    if (e.type == SDL_KEYDOWN)
    {
        if (e.key.keysym.sym == SDLK_LCTRL)
        {
            _dollyMotion = true;
            _accumDeltaX = 0.0f;
            _accumDeltaY = 0.0f;
        }
    }
    else if (e.type == SDL_KEYUP)
    {
        if (e.key.keysym.sym == SDLK_LCTRL)
        {
            _dollyMotion = false;
        }
    }

    if (e.type == SDL_MOUSEMOTION)
    {
        const float sens = 0.1f;

        if (_panningMotion || _orbitingMotion)
        {
            _accumDeltaX += (float)e.motion.xrel * sens;
            _accumDeltaY -= (float)e.motion.yrel * sens;
        }
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

glm::vec3 Camera::GetUp()
{
    glm::mat3 rotation = glm::mat3(getRotationMatrix());
    return rotation[1];
}


glm::vec3 Camera::GetRight()
{
    glm::mat3 rotation = glm::mat3(getRotationMatrix());
    return rotation[0];
}
