// Research-only instrumentation for pinned PyBullet 3.2.7. No production API.
// Included in PhysicsServerCommandProcessor.cpp; all ordinary stepping unchanged.
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <stdexcept>

struct SECacheIO {
    bool reading;
    int mask;
    std::fstream file;
    SECacheIO(const char* path, bool read, int fields): reading(read), mask(fields),
        file(path, read ? std::ios::in : std::ios::out) {
        if (!file) throw std::runtime_error("research cache file open failed");
        file << std::setprecision(17);
    }
    template<class T> void scalar(const std::string& key, T& value, int group) {
        if (reading) {
            std::string actual; double v;
            if (!(file >> actual >> v) || actual != key)
                throw std::runtime_error("research cache schema/topology mismatch");
            if (mask & group) value = static_cast<T>(v);
        } else file << key << " " << value << "\n";
    }
    void vector(const std::string& key, btVector3& value, int group) {
        for (int j=0; j<3; ++j) scalar(key+"/"+std::to_string(j), value[j], group);
    }
    void quaternion(const std::string& key, btQuaternion& value, int group) {
        for (int j=0; j<4; ++j) scalar(key+"/"+std::to_string(j), value[j], group);
    }
    void transform(const std::string& key, btTransform& value, int group) {
        for (int j=0; j<3; ++j) vector(key+"/basis/"+std::to_string(j), value.getBasis()[j], group);
        vector(key+"/origin", value.getOrigin(), group);
    }
};

void seRocketCacheHook(btMultiBodyDynamicsWorld* world) {
    const char* operation = std::getenv("SE_ROCKET_CACHE_OPERATION");
    const char* filename = std::getenv("SE_ROCKET_CACHE_FILE");
    if (!operation || !filename) return;
    bool read = std::string(operation) == "import";
    const char* fields = std::getenv("SE_ROCKET_CACHE_MASK");
    SECacheIO io(filename, read, fields ? std::atoi(fields) : 31);
    int count = world->getNumMultibodies();
    int expected = count;
    io.scalar("topology/multibodies", count, 127);
    if (count != expected) throw std::runtime_error("research multibody count mismatch");
    for (int b=0; b<count; ++b) {
        btMultiBody* body = world->getMultiBody(b);
        std::string prefix = "multibody/"+std::to_string(b);
        int links = body->getNumLinks(); expected = links;
        io.scalar(prefix+"/links", links, 127);
        if (links != expected) throw std::runtime_error("research link count mismatch");
        btVector3 base = body->getInterpolateBasePos();
        btQuaternion rotation = body->getInterpolateWorldToBaseRot();
        io.vector(prefix+"/interpolate_base_position", base, 8);
        io.quaternion(prefix+"/interpolate_world_to_base", rotation, 8);
        if (read && (io.mask & 8)) {
            body->setInterpolateBasePos(base);
            body->setInterpolateWorldToBaseRot(rotation);
        }
        for (int l=0; l<links; ++l) {
            btMultibodyLink& link = body->getLink(l);
            std::string key = prefix+"/link/"+std::to_string(l);
            io.transform(key+"/cached_world", link.m_cachedWorldTransform, 1);
            io.vector(key+"/total_velocity_angular", link.m_absFrameTotVelocity.m_topVec, 4);
            io.vector(key+"/total_velocity_linear", link.m_absFrameTotVelocity.m_bottomVec, 4);
            io.vector(key+"/local_velocity_angular", link.m_absFrameLocVelocity.m_topVec, 4);
            io.vector(key+"/local_velocity_linear", link.m_absFrameLocVelocity.m_bottomVec, 4);
            io.vector(key+"/cached_r", link.m_cachedRVector, 8);
            io.quaternion(key+"/cached_parent_rotation", link.m_cachedRotParentToThis, 8);
            io.vector(key+"/interpolate_r", link.m_cachedRVector_interpolate, 8);
            io.quaternion(key+"/interpolate_parent_rotation", link.m_cachedRotParentToThis_interpolate, 8);
            for (int j=0; j<link.m_posVarCount; ++j)
                io.scalar(key+"/interpolate_joint/"+std::to_string(j), link.m_jointPos_interpolate[j], 8);
        }
    }
    count = world->getNumCollisionObjects(); expected = count;
    io.scalar("topology/colliders", count, 127);
    if (count != expected) throw std::runtime_error("research collider count mismatch");
    for (int c=0; c<count; ++c) {
        btCollisionObject* collider = world->getCollisionObjectArray()[c];
        std::string key = "collider/"+std::to_string(c);
        btTransform transform = collider->getWorldTransform();
        btTransform interpolation = collider->getInterpolationWorldTransform();
        btVector3 linear = collider->getInterpolationLinearVelocity();
        btVector3 angular = collider->getInterpolationAngularVelocity();
        io.transform(key+"/world", transform, 2);
        io.transform(key+"/interpolation", interpolation, 2);
        io.vector(key+"/interpolation_linear_velocity", linear, 2);
        io.vector(key+"/interpolation_angular_velocity", angular, 2);
        if (read && (io.mask & 2)) {
            collider->setWorldTransform(transform);
            collider->setInterpolationWorldTransform(interpolation);
            collider->setInterpolationLinearVelocity(linear);
            collider->setInterpolationAngularVelocity(angular);
        }
        int activation = collider->getActivationState();
        btScalar deactivation = collider->getDeactivationTime();
        io.scalar(key+"/activation", activation, 16);
        io.scalar(key+"/deactivation_time", deactivation, 16);
        if (read && (io.mask & 16)) {
            collider->forceActivationState(activation);
            collider->setDeactivationTime(deactivation);
        }
    }
    if (!read) {
        std::string contactFile = std::string(filename) + ".contacts";
        SECacheIO contacts(contactFile.c_str(), false, 0);
        int manifolds = world->getDispatcher()->getNumManifolds();
        contacts.scalar("manifolds/count", manifolds, 0);
        for (int m=0; m<manifolds; ++m) {
            btPersistentManifold* manifold = world->getDispatcher()->getManifoldByIndexInternal(m);
            std::string key = "manifold/"+std::to_string(m);
            int a = manifold->getBody0()->getBroadphaseHandle()->m_uniqueId;
            int b = manifold->getBody1()->getBroadphaseHandle()->m_uniqueId;
            int points = manifold->getNumContacts();
            contacts.scalar(key+"/diagnostic_broadphase_a", a, 0);
            contacts.scalar(key+"/diagnostic_broadphase_b", b, 0);
            contacts.scalar(key+"/points", points, 0);
            for (int c=0; c<points; ++c) {
                btManifoldPoint& point = manifold->getContactPoint(c);
                std::string row = key+"/point/"+std::to_string(c);
                contacts.vector(row+"/m_localPointA", point.m_localPointA, 0);
                contacts.vector(row+"/m_localPointB", point.m_localPointB, 0);
                contacts.vector(row+"/m_positionWorldOnA", point.m_positionWorldOnA, 0);
                contacts.vector(row+"/m_positionWorldOnB", point.m_positionWorldOnB, 0);
                contacts.vector(row+"/m_normalWorldOnB", point.m_normalWorldOnB, 0);
                contacts.vector(row+"/m_lateralFrictionDir1", point.m_lateralFrictionDir1, 0);
                contacts.vector(row+"/m_lateralFrictionDir2", point.m_lateralFrictionDir2, 0);
                contacts.scalar(row+"/m_distance1", point.m_distance1, 0);
                contacts.scalar(row+"/m_combinedFriction", point.m_combinedFriction, 0);
                contacts.scalar(row+"/m_combinedRollingFriction", point.m_combinedRollingFriction, 0);
                contacts.scalar(row+"/m_combinedSpinningFriction", point.m_combinedSpinningFriction, 0);
                contacts.scalar(row+"/m_combinedRestitution", point.m_combinedRestitution, 0);
                contacts.scalar(row+"/m_partId0", point.m_partId0, 0);
                contacts.scalar(row+"/m_partId1", point.m_partId1, 0);
                contacts.scalar(row+"/m_index0", point.m_index0, 0);
                contacts.scalar(row+"/m_index1", point.m_index1, 0);
                contacts.scalar(row+"/m_contactPointFlags", point.m_contactPointFlags, 0);
                contacts.scalar(row+"/m_appliedImpulse", point.m_appliedImpulse, 0);
                contacts.scalar(row+"/m_prevRHS", point.m_prevRHS, 0);
                contacts.scalar(row+"/m_appliedImpulseLateral1", point.m_appliedImpulseLateral1, 0);
                contacts.scalar(row+"/m_appliedImpulseLateral2", point.m_appliedImpulseLateral2, 0);
                contacts.scalar(row+"/m_contactMotion1", point.m_contactMotion1, 0);
                contacts.scalar(row+"/m_contactMotion2", point.m_contactMotion2, 0);
                contacts.scalar(row+"/m_contactCFM", point.m_contactCFM, 0);
                contacts.scalar(row+"/m_contactERP", point.m_contactERP, 0);
                contacts.scalar(row+"/m_frictionCFM", point.m_frictionCFM, 0);
                contacts.scalar(row+"/m_lifeTime", point.m_lifeTime, 0);
            }
        }
    }

}
