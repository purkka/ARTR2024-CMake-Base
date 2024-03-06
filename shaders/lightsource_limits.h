#ifndef LIGHTSOURCE_LIMITS_H

// By default we create 1 ambient, 1 directional and 98 point lights = 100 lights in total.
// If you have an ultra-fast GPU and can not see any artefacts when preparing to tackle Task 3, you can add some extra point lights.
// For example, if you change the 0 in the line below to 100, you will get 100 extra point lights, making a total of 200 lights.

#define EXTRA_POINTLIGHTS	0


// --- don't touch anything below ---

#if EXTRA_POINTLIGHTS < 28
#define MAX_NUMBER_OF_LIGHTSOURCES 128
#else
#define MAX_NUMBER_OF_LIGHTSOURCES (100 + EXTRA_POINTLIGHTS)
#endif

#define LIGHTSOURCE_LIMITS_H 1
#endif

