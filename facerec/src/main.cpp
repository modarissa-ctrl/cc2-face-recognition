/**
 * @file main.cpp
 * @brief Desktop entry point and headless utility modes for `facerec`.
 */

#include "ofMain.h"
#include "ofApp.h"
#include "HeadlessCommands.h"
#include "HeadlessSelftest.h"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++)
    {
        args.push_back(argv[i]);
    }

    // Execute CLI-only modes before touching any GL state so CI and remote
    // hosts can run diagnostics without a display server.
    if (std::find(args.begin(), args.end(), "--selftest") != args.end())
    {
        return headlessSelftest::runSelftest();
    }
    // Diagnostic video replay: --liveness-replay <video> [fps]. Kept separate
    // from the single-image modes below because it takes an optional sampling
    // rate after the required path.
    if (auto flagIt = std::find(args.begin(), args.end(), "--liveness-replay"); flagIt != args.end())
    {
        auto pathIt = std::next(flagIt);
        if (pathIt == args.end() || (!pathIt->empty() && (*pathIt)[0] == '-'))
        {
            std::fprintf(stderr, "usage: facerec --liveness-replay <video> [fps]\n");
            return 1;
        }
        float targetFps = 23.0f;
        if (auto fpsIt = std::next(pathIt); fpsIt != args.end() && !fpsIt->empty() && (*fpsIt)[0] != '-')
        {
            targetFps = std::stof(*fpsIt);
        }
        return headlessCommands::runHeadlessLivenessReplay(*pathIt, targetFps);
    }
    for (auto [flag, run] : {std::pair{"--detect", headlessCommands::runHeadlessDetect},
                             std::pair{"--identify", headlessCommands::runHeadlessIdentify}})
    {
        auto flagIt = std::find(args.begin(), args.end(), flag);
        if (flagIt == args.end())
        {
            continue;
        }
        auto imageIt = std::next(flagIt);
        if (imageIt == args.end() || (!imageIt->empty() && (*imageIt)[0] == '-'))
        {
            std::fprintf(stderr, "usage: facerec %s <image>\n", flag);
            return 1;
        }
        return run(*imageIt);
    }

    ofGLWindowSettings settings;
    settings.setSize(1024, 768);
    settings.title = "facerec";

    auto window = ofCreateWindow(settings);
    auto app = std::make_shared<ofApp>();
    app->args = args;
    ofRunApp(window, app);
    return ofRunMainLoop();
}
