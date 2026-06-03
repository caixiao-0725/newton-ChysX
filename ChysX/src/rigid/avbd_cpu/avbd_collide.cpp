// SPDX-License-Identifier: MIT
// OBB-OBB SAT collision detection. Adapted from avbd-demo3d.

#include "avbd_solver.h"

#include <cfloat>
#include <cmath>

namespace chysx {
namespace avbd {

namespace {

constexpr int MAX_CONTACTS = 8;
constexpr int MAX_POLY_VERTS = 16;
constexpr float SAT_AXIS_EPSILON = 1.0e-6f;
constexpr float PLANE_EPSILON = 1.0e-5f;
constexpr float CONTACT_MERGE_DIST_SQ = 1.0e-6f;

enum AxisType { AXIS_FACE_A, AXIS_FACE_B, AXIS_EDGE };

struct OBB {
    float3 center;
    quat rotation;
    float3 half;
};

struct SatAxis {
    AxisType type;
    int indexA;
    int indexB;
    float separation;
    float3 normalAB;
    bool valid;
};

struct FaceFrame {
    int axisIndex;
    float3 normal;
    float3 center;
    float3 u;
    float3 v;
    float extentU;
    float extentV;
};

inline OBB makeOBB(const Rigid* body) {
    OBB box{};
    box.center = body->positionLin;
    box.rotation = body->positionAng;
    box.half = body->size * 0.5f;
    return box;
}

inline void computeAxes(const OBB& box, float3 ax[3]) {
    ax[0] = rotate(box.rotation, float3{1.0f, 0.0f, 0.0f});
    ax[1] = rotate(box.rotation, float3{0.0f, 1.0f, 0.0f});
    ax[2] = rotate(box.rotation, float3{0.0f, 0.0f, 1.0f});
}

inline float absDot(float3 a, float3 b) { return std::fabs(dot(a, b)); }

inline float3 supportPoint(const OBB& box, const float3 ax[3], const float3& dir) {
    float sx = dot(dir, ax[0]) >= 0.0f ? 1.0f : -1.0f;
    float sy = dot(dir, ax[1]) >= 0.0f ? 1.0f : -1.0f;
    float sz = dot(dir, ax[2]) >= 0.0f ? 1.0f : -1.0f;
    return box.center + ax[0] * (box.half.x * sx) +
           ax[1] * (box.half.y * sy) + ax[2] * (box.half.z * sz);
}

inline void getFaceAxes(const float3 ax[3], const float3& halfExt, int axisIndex,
                        float3& u, float3& v, float& extentU, float& extentV) {
    if (axisIndex == 0) {
        u = ax[1]; v = ax[2]; extentU = halfExt.y; extentV = halfExt.z;
    } else if (axisIndex == 1) {
        u = ax[0]; v = ax[2]; extentU = halfExt.x; extentV = halfExt.z;
    } else {
        u = ax[0]; v = ax[1]; extentU = halfExt.x; extentV = halfExt.y;
    }
}

inline void buildFaceFrame(const OBB& box, const float3 ax[3], int axisIndex,
                           const float3& outwardNormal, FaceFrame& frame) {
    float s = dot(outwardNormal, ax[axisIndex]) >= 0.0f ? 1.0f : -1.0f;
    frame.axisIndex = axisIndex;
    frame.normal = ax[axisIndex] * s;
    frame.center = box.center + frame.normal * box.half[axisIndex];
    getFaceAxes(ax, box.half, axisIndex, frame.u, frame.v, frame.extentU, frame.extentV);
}

inline int chooseIncidentFaceAxis(const float3 ax[3], const float3& referenceNormal) {
    int best_axis = 0;
    float best = -FLT_MAX;
    for (int i = 0; i < 3; ++i) {
        float d = absDot(ax[i], referenceNormal);
        if (d > best) { best = d; best_axis = i; }
    }
    return best_axis;
}

inline void buildIncidentFace(const OBB& box, const float3 ax[3], int axisIndex,
                              const float3& referenceNormal, float3 outVerts[4]) {
    float s = dot(ax[axisIndex], referenceNormal) > 0.0f ? -1.0f : 1.0f;
    float3 faceNormal = ax[axisIndex] * s;
    float3 faceCenter = box.center + faceNormal * box.half[axisIndex];
    float3 u, v; float extU, extV;
    getFaceAxes(ax, box.half, axisIndex, u, v, extU, extV);
    outVerts[0] = faceCenter + u * extU + v * extV;
    outVerts[1] = faceCenter - u * extU + v * extV;
    outVerts[2] = faceCenter - u * extU - v * extV;
    outVerts[3] = faceCenter + u * extU - v * extV;
}

inline int clipPolygonAgainstPlane(const float3* inVerts, int inCount,
                                   const float3& planeNormal, float planeOffset,
                                   float3* outVerts) {
    if (inCount <= 0) return 0;
    int outCount = 0;
    float3 a = inVerts[inCount - 1];
    float da = dot(planeNormal, a) - planeOffset;
    for (int i = 0; i < inCount; ++i) {
        float3 b = inVerts[i];
        float db = dot(planeNormal, b) - planeOffset;
        bool aInside = da <= PLANE_EPSILON;
        bool bInside = db <= PLANE_EPSILON;
        if (aInside != bInside) {
            float t = 0.0f;
            float denom = da - db;
            if (std::fabs(denom) > SAT_AXIS_EPSILON)
                t = clamp(da / denom, 0.0f, 1.0f);
            if (outCount < MAX_POLY_VERTS)
                outVerts[outCount++] = a + (b - a) * t;
        }
        if (bInside && outCount < MAX_POLY_VERTS)
            outVerts[outCount++] = b;
        a = b;
        da = db;
    }
    return outCount;
}

inline bool addContact(Rigid* bodyA, Rigid* bodyB,
                       Manifold::Contact* contacts, int& contactCount,
                       float3* contactMidpoints,
                       float3 xA, float3 xB, int featureKey) {
    float3 midpoint = (xA + xB) * 0.5f;
    for (int i = 0; i < contactCount; ++i) {
        float3 d = midpoint - contactMidpoints[i];
        if (lengthSq(d) < CONTACT_MERGE_DIST_SQ) return false;
    }
    if (contactCount >= MAX_CONTACTS) return false;

    Manifold::FeaturePair feature{};
    feature.key = featureKey;

    Manifold::Contact& c = contacts[contactCount];
    c.feature = feature;
    c.rA = rotate(conjugate(bodyA->positionAng), xA - bodyA->positionLin);
    c.rB = rotate(conjugate(bodyB->positionAng), xB - bodyB->positionLin);
    contactMidpoints[contactCount] = midpoint;
    ++contactCount;
    return true;
}

inline bool testAxis(const OBB& boxA, const float3 axA[3],
                     const OBB& boxB, const float3 axB[3],
                     const float3& delta, const float3& axis,
                     AxisType type, int indexA, int indexB, SatAxis& best) {
    float lenSq = lengthSq(axis);
    if (lenSq < SAT_AXIS_EPSILON) return true;
    float invLen = 1.0f / std::sqrt(lenSq);
    float3 n = axis * invLen;
    if (dot(n, delta) < 0.0f) n = -n;
    float distance = std::fabs(dot(delta, n));
    float rA = boxA.half.x * absDot(n, axA[0]) +
               boxA.half.y * absDot(n, axA[1]) +
               boxA.half.z * absDot(n, axA[2]);
    float rB = boxB.half.x * absDot(n, axB[0]) +
               boxB.half.y * absDot(n, axB[1]) +
               boxB.half.z * absDot(n, axB[2]);
    float separation = distance - (rA + rB);
    if (separation > 0.0f) return false;
    if (!best.valid || separation > best.separation) {
        best.valid = true;
        best.type = type;
        best.indexA = indexA;
        best.indexB = indexB;
        best.separation = separation;
        best.normalAB = n;
    }
    return true;
}

inline void supportEdge(const OBB& box, const float3 ax[3], int axisIndex,
                        const float3& dir, float3& edgeA, float3& edgeB) {
    int axis1 = (axisIndex + 1) % 3;
    int axis2 = (axisIndex + 2) % 3;
    float sign1 = dot(dir, ax[axis1]) >= 0.0f ? 1.0f : -1.0f;
    float sign2 = dot(dir, ax[axis2]) >= 0.0f ? 1.0f : -1.0f;
    float3 edgeCenter = box.center +
        ax[axis1] * (box.half[axis1] * sign1) +
        ax[axis2] * (box.half[axis2] * sign2);
    edgeA = edgeCenter - ax[axisIndex] * box.half[axisIndex];
    edgeB = edgeCenter + ax[axisIndex] * box.half[axisIndex];
}

inline void closestPointsOnSegments(const float3& p0, const float3& p1,
                                    const float3& q0, const float3& q1,
                                    float3& c0, float3& c1) {
    float3 d1 = p1 - p0;
    float3 d2 = q1 - q0;
    float3 r = p0 - q0;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;
    if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON) { c0 = p0; c1 = q0; return; }
    if (a <= SAT_AXIS_EPSILON) {
        t = clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= SAT_AXIS_EPSILON) {
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            if (std::fabs(denom) > SAT_AXIS_EPSILON)
                s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            t = (b * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = clamp((b - c) / a, 0.0f, 1.0f); }
        }
    }
    c0 = p0 + d1 * s;
    c1 = q0 + d2 * t;
}

inline int buildFaceManifold(Rigid* bodyA, Rigid* bodyB,
                             const OBB& boxA, const float3 axA[3],
                             const OBB& boxB, const float3 axB[3],
                             bool referenceIsA, int referenceAxis,
                             const float3& normalAB,
                             Manifold::Contact* contacts) {
    const OBB& referenceBox = referenceIsA ? boxA : boxB;
    const float3* refAx = referenceIsA ? axA : axB;
    const OBB& incidentBox = referenceIsA ? boxB : boxA;
    const float3* incAx = referenceIsA ? axB : axA;
    float3 referenceOutward = referenceIsA ? normalAB : -normalAB;

    FaceFrame referenceFace{};
    buildFaceFrame(referenceBox, refAx, referenceAxis, referenceOutward, referenceFace);
    int incidentAxis = chooseIncidentFaceAxis(incAx, referenceFace.normal);

    float3 clip0[MAX_POLY_VERTS];
    float3 clip1[MAX_POLY_VERTS];
    buildIncidentFace(incidentBox, incAx, incidentAxis, referenceFace.normal, clip0);
    int count = 4;

    count = clipPolygonAgainstPlane(clip0, count, referenceFace.u,
                                   dot(referenceFace.u, referenceFace.center) + referenceFace.extentU, clip1);
    if (!count) return 0;
    count = clipPolygonAgainstPlane(clip1, count, -referenceFace.u,
                                   dot(-referenceFace.u, referenceFace.center) + referenceFace.extentU, clip0);
    if (!count) return 0;
    count = clipPolygonAgainstPlane(clip0, count, referenceFace.v,
                                   dot(referenceFace.v, referenceFace.center) + referenceFace.extentV, clip1);
    if (!count) return 0;
    count = clipPolygonAgainstPlane(clip1, count, -referenceFace.v,
                                   dot(-referenceFace.v, referenceFace.center) + referenceFace.extentV, clip0);
    if (!count) return 0;

    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    int featurePrefix = (referenceIsA ? AXIS_FACE_A : AXIS_FACE_B) << 24;
    featurePrefix |= (referenceAxis & 0xFF) << 16;
    featurePrefix |= (incidentAxis & 0xFF) << 8;

    for (int i = 0; i < count && contactCount < MAX_CONTACTS; ++i) {
        float3 pIncident = clip0[i];
        float distance = dot(pIncident - referenceFace.center, referenceFace.normal);
        if (distance > PLANE_EPSILON) continue;
        float3 pReference = pIncident - referenceFace.normal * distance;
        float3 xA = referenceIsA ? pReference : pIncident;
        float3 xB = referenceIsA ? pIncident : pReference;
        addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints,
                   xA, xB, featurePrefix | (i & 0xFF));
    }

    if (!contactCount) {
        float3 xA = supportPoint(boxA, axA, normalAB);
        float3 xB = supportPoint(boxB, axB, -normalAB);
        addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featurePrefix);
    }
    return contactCount;
}

inline int buildEdgeContact(Rigid* bodyA, Rigid* bodyB,
                            const OBB& boxA, const float3 axA[3],
                            const OBB& boxB, const float3 axB[3],
                            int axisA, int axisB, const float3& normalAB,
                            Manifold::Contact* contacts) {
    float3 a0, a1, b0, b1;
    supportEdge(boxA, axA, axisA, normalAB, a0, a1);
    supportEdge(boxB, axB, axisB, -normalAB, b0, b1);
    float3 xA, xB;
    closestPointsOnSegments(a0, a1, b0, b1, xA, xB);
    int contactCount = 0;
    float3 contactMidpoints[MAX_CONTACTS];
    int featureKey = (AXIS_EDGE << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
    addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featureKey);
    if (!contactCount) {
        xA = supportPoint(boxA, axA, normalAB);
        xB = supportPoint(boxB, axB, -normalAB);
        addContact(bodyA, bodyB, contacts, contactCount, contactMidpoints, xA, xB, featureKey);
    }
    return contactCount;
}

}  // namespace

int Manifold::collide(Rigid* bodyA, Rigid* bodyB, Contact* contacts, float3x3& basisOut) {
    OBB boxA = makeOBB(bodyA);
    OBB boxB = makeOBB(bodyB);

    float3 axA[3], axB[3];
    computeAxes(boxA, axA);
    computeAxes(boxB, axB);

    float3 delta = boxB.center - boxA.center;

    SatAxis bestFace{};
    bestFace.separation = -FLT_MAX;
    bestFace.valid = false;

    SatAxis bestEdge{};
    bestEdge.separation = -FLT_MAX;
    bestEdge.valid = false;

    for (int i = 0; i < 3; ++i)
        if (!testAxis(boxA, axA, boxB, axB, delta, axA[i], AXIS_FACE_A, i, -1, bestFace)) return 0;
    for (int i = 0; i < 3; ++i)
        if (!testAxis(boxA, axA, boxB, axB, delta, axB[i], AXIS_FACE_B, -1, i, bestFace)) return 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!testAxis(boxA, axA, boxB, axB, delta, cross(axA[i], axB[j]),
                          AXIS_EDGE, i, j, bestEdge)) return 0;

    if (!bestFace.valid) return 0;

    SatAxis best = bestFace;
    if (bestEdge.valid) {
        const float edgeRelTol = 0.95f;
        const float edgeAbsTol = 0.01f;
        if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol)
            best = bestEdge;
    }

    basisOut = orthonormal(-best.normalAB);

    if (best.type == AXIS_EDGE)
        return buildEdgeContact(bodyA, bodyB, boxA, axA, boxB, axB, best.indexA, best.indexB, best.normalAB, contacts);
    if (best.type == AXIS_FACE_A)
        return buildFaceManifold(bodyA, bodyB, boxA, axA, boxB, axB, true, best.indexA, best.normalAB, contacts);
    return buildFaceManifold(bodyA, bodyB, boxA, axA, boxB, axB, false, best.indexB, best.normalAB, contacts);
}

}  // namespace avbd
}  // namespace chysx
