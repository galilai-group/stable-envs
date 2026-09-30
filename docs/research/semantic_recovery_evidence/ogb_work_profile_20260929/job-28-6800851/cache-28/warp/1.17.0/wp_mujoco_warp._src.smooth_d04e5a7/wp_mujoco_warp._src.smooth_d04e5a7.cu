#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

// Maximum registers per thread. __maxnreg__ was added in CUDA Toolkit 12.4;
// older toolkits ignore the opt-in so the same source remains compilable.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define WP_MAXNREG(n) __maxnreg__(n)
#else
#define WP_MAXNREG(n)
#endif

// Allow PTXAS to spill registers into shared memory. CUDA Toolkit 13.0
// introduced the pragma, which is unavailable in device-debug compilation.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13) && !defined(_DEBUG)
#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \"enable_smem_spilling\";");
#else
#define WP_ENABLE_SMEM_SPILLING()
#endif


// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:44
static CUDA_CALLABLE wp::vec_t<3, wp::float32> rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    const wp::float32 var_9 = 2.0;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::float32 var_18 = 2.0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def rot_vec_quat(vec: wp.vec3, quat: wp.quat) -> wp.vec3:                              <L 45>
    // s, u = quat[0], wp.vec3(quat[1], quat[2], quat[3])                                     <L 46>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_7 = wp::extract(var_quat, var_6);
    var_8 = wp::vec_t<3, wp::float32>(var_3, var_5, var_7);
    // r = 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec                          <L 47>
    var_10 = wp::dot(var_8, var_vec);
    var_11 = wp::mul(var_10, var_8);
    var_12 = wp::mul(var_9, var_11);
    var_13 = wp::mul(var_1, var_1);
    var_14 = wp::dot(var_8, var_8);
    var_15 = wp::sub(var_13, var_14);
    var_16 = wp::mul(var_15, var_vec);
    var_17 = wp::add(var_12, var_16);
    // r = r + 2.0 * s * wp.cross(u, vec)                                                     <L 48>
    var_19 = wp::mul(var_18, var_1);
    var_20 = wp::cross(var_8, var_vec);
    var_21 = wp::mul(var_19, var_20);
    var_22 = wp::add(var_17, var_21);
    // return r                                                                               <L 49>
    return var_22;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE wp::quat_t<wp::float32> mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 3;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    const wp::int32 var_53 = 3;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 3;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 3;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    wp::float32 var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 3;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::quat_t<wp::float32> var_92;
    //---------
    // forward
    // def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:                                       <L 24>
    // return wp.quat(                                                                        <L 25>
    // u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],                                 <L 26>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_u, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_u, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    // u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],                                 <L 27>
    var_24 = wp::extract(var_u, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_u, var_28);
    var_31 = wp::extract(var_v, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::add(var_27, var_32);
    var_35 = wp::extract(var_u, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_u, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    // u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],                                 <L 28>
    var_47 = wp::extract(var_u, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_52 = wp::extract(var_u, var_51);
    var_54 = wp::extract(var_v, var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::sub(var_50, var_55);
    var_58 = wp::extract(var_u, var_57);
    var_60 = wp::extract(var_v, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_u, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    // u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],                                 <L 29>
    var_70 = wp::extract(var_u, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_75 = wp::extract(var_u, var_74);
    var_77 = wp::extract(var_v, var_76);
    var_78 = wp::mul(var_75, var_77);
    var_79 = wp::add(var_73, var_78);
    var_81 = wp::extract(var_u, var_80);
    var_83 = wp::extract(var_v, var_82);
    var_84 = wp::mul(var_81, var_83);
    var_85 = wp::sub(var_79, var_84);
    var_87 = wp::extract(var_u, var_86);
    var_89 = wp::extract(var_v, var_88);
    var_90 = wp::mul(var_87, var_89);
    var_91 = wp::add(var_85, var_90);
    var_92 = wp::quat_t<wp::float32>(var_22, var_45, var_68, var_91);
    return var_92;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:59
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> quat_to_mat_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    const wp::int32 var_32 = 3;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    const wp::int32 var_42 = 3;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 3;
    wp::float32 var_46;
    const wp::int32 var_47 = 3;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 2.0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 2.0;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 2.0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::float32 var_65 = 2.0;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 2.0;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 2.0;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::mat_t<3, 3, wp::float32> var_77;
    //---------
    // forward
    // def quat_to_mat(quat: wp.quat) -> wp.mat33:                                            <L 60>
    // q00 = quat[0] * quat[0]                                                                <L 62>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::mul(var_1, var_3);
    // q01 = quat[0] * quat[1]                                                                <L 63>
    var_6 = wp::extract(var_quat, var_5);
    var_8 = wp::extract(var_quat, var_7);
    var_9 = wp::mul(var_6, var_8);
    // q02 = quat[0] * quat[2]                                                                <L 64>
    var_11 = wp::extract(var_quat, var_10);
    var_13 = wp::extract(var_quat, var_12);
    var_14 = wp::mul(var_11, var_13);
    // q03 = quat[0] * quat[3]                                                                <L 65>
    var_16 = wp::extract(var_quat, var_15);
    var_18 = wp::extract(var_quat, var_17);
    var_19 = wp::mul(var_16, var_18);
    // q11 = quat[1] * quat[1]                                                                <L 66>
    var_21 = wp::extract(var_quat, var_20);
    var_23 = wp::extract(var_quat, var_22);
    var_24 = wp::mul(var_21, var_23);
    // q12 = quat[1] * quat[2]                                                                <L 67>
    var_26 = wp::extract(var_quat, var_25);
    var_28 = wp::extract(var_quat, var_27);
    var_29 = wp::mul(var_26, var_28);
    // q13 = quat[1] * quat[3]                                                                <L 68>
    var_31 = wp::extract(var_quat, var_30);
    var_33 = wp::extract(var_quat, var_32);
    var_34 = wp::mul(var_31, var_33);
    // q22 = quat[2] * quat[2]                                                                <L 69>
    var_36 = wp::extract(var_quat, var_35);
    var_38 = wp::extract(var_quat, var_37);
    var_39 = wp::mul(var_36, var_38);
    // q23 = quat[2] * quat[3]                                                                <L 70>
    var_41 = wp::extract(var_quat, var_40);
    var_43 = wp::extract(var_quat, var_42);
    var_44 = wp::mul(var_41, var_43);
    // q33 = quat[3] * quat[3]                                                                <L 71>
    var_46 = wp::extract(var_quat, var_45);
    var_48 = wp::extract(var_quat, var_47);
    var_49 = wp::mul(var_46, var_48);
    // return wp.mat33(                                                                       <L 73>
    // q00 + q11 - q22 - q33,                                                                 <L 74>
    var_50 = wp::add(var_4, var_24);
    var_51 = wp::sub(var_50, var_39);
    var_52 = wp::sub(var_51, var_49);
    // 2.0 * (q12 - q03),                                                                     <L 75>
    var_54 = wp::sub(var_29, var_19);
    var_55 = wp::mul(var_53, var_54);
    // 2.0 * (q13 + q02),                                                                     <L 76>
    var_57 = wp::add(var_34, var_14);
    var_58 = wp::mul(var_56, var_57);
    // 2.0 * (q12 + q03),                                                                     <L 77>
    var_60 = wp::add(var_29, var_19);
    var_61 = wp::mul(var_59, var_60);
    // q00 - q11 + q22 - q33,                                                                 <L 78>
    var_62 = wp::sub(var_4, var_24);
    var_63 = wp::add(var_62, var_39);
    var_64 = wp::sub(var_63, var_49);
    // 2.0 * (q23 - q01),                                                                     <L 79>
    var_66 = wp::sub(var_44, var_9);
    var_67 = wp::mul(var_65, var_66);
    // 2.0 * (q13 - q02),                                                                     <L 80>
    var_69 = wp::sub(var_34, var_14);
    var_70 = wp::mul(var_68, var_69);
    // 2.0 * (q23 + q01),                                                                     <L 81>
    var_72 = wp::add(var_44, var_9);
    var_73 = wp::mul(var_71, var_72);
    // q00 - q11 - q22 + q33,                                                                 <L 82>
    var_74 = wp::sub(var_4, var_24);
    var_75 = wp::sub(var_74, var_39);
    var_76 = wp::add(var_75, var_49);
    var_77 = wp::mat_t<3, 3, wp::float32>(var_52, var_55, var_58, var_61, var_64, var_67, var_70, var_73, var_76);
    return var_77;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:133
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:       <L 134>
    // u0 = wp.vec3(u[0], u[1], u[2])                                                         <L 136>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_u, var_2);
    var_5 = wp::extract(var_u, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // u1 = wp.vec3(u[3], u[4], u[5])                                                         <L 137>
    var_8 = wp::extract(var_u, var_7);
    var_10 = wp::extract(var_u, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 138>
    var_15 = wp::extract(var_v, var_14);
    var_17 = wp::extract(var_v, var_16);
    var_19 = wp::extract(var_v, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 139>
    var_22 = wp::extract(var_v, var_21);
    var_24 = wp::extract(var_v, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(u0, v0)                                                                 <L 141>
    var_28 = wp::cross(var_6, var_20);
    // vel = wp.cross(u1, v0) + wp.cross(u0, v1)                                              <L 142>
    var_29 = wp::cross(var_13, var_20);
    var_30 = wp::cross(var_6, var_27);
    var_31 = wp::add(var_29, var_30);
    // return wp.spatial_vector(ang, vel)                                                     <L 144>
    var_32 = wp::vec_t<6, wp::float32>(var_28, var_31);
    return var_32;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:120
static CUDA_CALLABLE wp::vec_t<6, wp::float32> inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 3;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 4;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 8;
    wp::float32 var_18;
    const wp::int32 var_19 = 4;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 7;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 3;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 5;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 8;
    wp::float32 var_47;
    const wp::int32 var_48 = 3;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 6;
    wp::float32 var_53;
    const wp::int32 var_54 = 5;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 4;
    wp::float32 var_59;
    const wp::int32 var_60 = 0;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 5;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 7;
    wp::float32 var_76;
    const wp::int32 var_77 = 3;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 6;
    wp::float32 var_82;
    const wp::int32 var_83 = 4;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 8;
    wp::float32 var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 7;
    wp::float32 var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 9;
    wp::float32 var_99;
    const wp::int32 var_100 = 3;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 6;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 8;
    wp::float32 var_110;
    const wp::int32 var_111 = 0;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 9;
    wp::float32 var_116;
    const wp::int32 var_117 = 4;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 7;
    wp::float32 var_122;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 6;
    wp::float32 var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 9;
    wp::float32 var_133;
    const wp::int32 var_134 = 5;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::vec_t<6, wp::float32> var_138;
    //---------
    // forward
    // def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:              <L 121>
    // return wp.spatial_vector(                                                              <L 123>
    // i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],                   <L 124>
    var_1 = wp::extract(var_i, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_i, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    var_12 = wp::extract(var_i, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_i, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    var_24 = wp::extract(var_i, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    // i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],                   <L 125>
    var_30 = wp::extract(var_i, var_29);
    var_32 = wp::extract(var_v, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_35 = wp::extract(var_i, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_i, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::add(var_39, var_44);
    var_47 = wp::extract(var_i, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::add(var_45, var_50);
    var_53 = wp::extract(var_i, var_52);
    var_55 = wp::extract(var_v, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_57 = wp::sub(var_51, var_56);
    // i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],                   <L 126>
    var_59 = wp::extract(var_i, var_58);
    var_61 = wp::extract(var_v, var_60);
    var_62 = wp::mul(var_59, var_61);
    var_64 = wp::extract(var_i, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    var_70 = wp::extract(var_i, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_74 = wp::add(var_68, var_73);
    var_76 = wp::extract(var_i, var_75);
    var_78 = wp::extract(var_v, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::sub(var_74, var_79);
    var_82 = wp::extract(var_i, var_81);
    var_84 = wp::extract(var_v, var_83);
    var_85 = wp::mul(var_82, var_84);
    var_86 = wp::add(var_80, var_85);
    // i[8] * v[1] - i[7] * v[2] + i[9] * v[3],                                               <L 127>
    var_88 = wp::extract(var_i, var_87);
    var_90 = wp::extract(var_v, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_93 = wp::extract(var_i, var_92);
    var_95 = wp::extract(var_v, var_94);
    var_96 = wp::mul(var_93, var_95);
    var_97 = wp::sub(var_91, var_96);
    var_99 = wp::extract(var_i, var_98);
    var_101 = wp::extract(var_v, var_100);
    var_102 = wp::mul(var_99, var_101);
    var_103 = wp::add(var_97, var_102);
    // i[6] * v[2] - i[8] * v[0] + i[9] * v[4],                                               <L 128>
    var_105 = wp::extract(var_i, var_104);
    var_107 = wp::extract(var_v, var_106);
    var_108 = wp::mul(var_105, var_107);
    var_110 = wp::extract(var_i, var_109);
    var_112 = wp::extract(var_v, var_111);
    var_113 = wp::mul(var_110, var_112);
    var_114 = wp::sub(var_108, var_113);
    var_116 = wp::extract(var_i, var_115);
    var_118 = wp::extract(var_v, var_117);
    var_119 = wp::mul(var_116, var_118);
    var_120 = wp::add(var_114, var_119);
    // i[7] * v[0] - i[6] * v[1] + i[9] * v[5],                                               <L 129>
    var_122 = wp::extract(var_i, var_121);
    var_124 = wp::extract(var_v, var_123);
    var_125 = wp::mul(var_122, var_124);
    var_127 = wp::extract(var_i, var_126);
    var_129 = wp::extract(var_v, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::sub(var_125, var_130);
    var_133 = wp::extract(var_i, var_132);
    var_135 = wp::extract(var_v, var_134);
    var_136 = wp::mul(var_133, var_135);
    var_137 = wp::add(var_131, var_136);
    var_138 = wp::vec_t<6, wp::float32>({var_28, var_57, var_86, var_103, var_120, var_137});
    return var_138;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<2, wp::float32> var_x,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<2, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<2, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:202
static CUDA_CALLABLE void wrap_inside_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::float32 var_radius,
    wp::int32 var_maxiter,
    wp::float32 var_zinit,
    wp::float32 var_tolerance,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    wp::vec_t<2, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::float32 var_13;
    bool var_14;
    bool var_15;
    bool var_16;
    const wp::float32 var_17 = 1e-15;
    bool var_18;
    bool var_19;
    bool var_20;
    const wp::float32 var_21 = -1.0;
    const wp::float32 var_22 = 10000000000.0;
    wp::vec_t<2, wp::float32> var_23;
    wp::vec_t<2, wp::float32> var_24;
    bool var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    bool var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    const wp::float32 var_32 = 1.0;
    bool var_33;
    wp::vec_t<2, wp::float32> var_34;
    wp::vec_t<2, wp::float32> var_35;
    wp::float32 var_36;
    bool var_37;
    const wp::float32 var_38 = -1.0;
    wp::vec_t<2, wp::float32> var_39;
    wp::vec_t<2, wp::float32> var_40;
    const wp::float32 var_41 = 0.5;
    wp::vec_t<2, wp::float32> var_42;
    wp::vec_t<2, wp::float32> var_43;
    wp::vec_t<2, wp::float32> var_44;
    wp::float32 var_45;
    wp::vec_t<2, wp::float32> var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 2.0;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = -1.0;
    wp::float32 var_60;
    bool var_61;
    const wp::float32 var_62 = -1.0;
    const wp::float32 var_63 = 1.0;
    wp::float32 var_64;
    bool var_65;
    const wp::float32 var_66 = 0.0;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::float32 var_74 = 2.0;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::float32 var_79 = 0.0;
    bool var_80;
    const wp::float32 var_81 = 0.0;
    const wp::int32 var_82 = 0;
    wp::int32 var_83;
    bool var_84;
    bool var_85;
    wp::float32 var_86;
    bool var_87;
    wp::float32 var_88;
    const wp::float32 var_89 = 1.0;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    const wp::float32 var_95 = 1.0;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    const wp::float32 var_102 = 2.0;
    const wp::float32 var_103 = 1.0;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::float32 var_109 = -1e-15;
    bool var_110;
    const wp::float32 var_111 = 0.0;
    wp::float32 var_112;
    wp::float32 var_113;
    bool var_114;
    const wp::float32 var_115 = 0.0;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::float32 var_122 = 2.0;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    wp::float32 var_126;
    bool var_127;
    const wp::float32 var_128 = 0.0;
    const wp::int32 var_129 = 1;
    wp::int32 var_130;
    bool var_131;
    const wp::float32 var_132 = 0.0;
    const wp::int32 var_133 = 0;
    wp::float32 var_134;
    const wp::int32 var_135 = 3;
    wp::float32 var_136;
    wp::float32 var_137;
    const wp::int32 var_138 = 1;
    wp::float32 var_139;
    const wp::int32 var_140 = 2;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    const wp::float32 var_144 = 0.0;
    bool var_145;
    wp::vec_t<2, wp::float32> var_146;
    wp::float32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    wp::vec_t<2, wp::float32> var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::vec_t<2, wp::float32> var_156;
    wp::float32 var_157;
    wp::vec_t<2, wp::float32> var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    const wp::int32 var_161 = 0;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    const wp::int32 var_165 = 1;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::float32 var_170;
    const wp::int32 var_171 = 0;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    const wp::int32 var_175 = 1;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    wp::vec_t<2, wp::float32> var_180;
    const wp::float32 var_181 = 0.0;
    //---------
    // forward
    // def wrap_inside(                                                                       <L 203>
    // end0 = wp.vec2(end[0], end[1])                                                         <L 224>
    var_1 = wp::extract(var_end, var_0);
    var_3 = wp::extract(var_end, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // end1 = wp.vec2(end[2], end[3])                                                         <L 225>
    var_6 = wp::extract(var_end, var_5);
    var_8 = wp::extract(var_end, var_7);
    var_9 = wp::vec_t<2, wp::float32>(var_6, var_8);
    // len0 = wp.norm_l2(end0)                                                                <L 228>
    var_10 = norm_l2_0(var_4);
    // len1 = wp.norm_l2(end1)                                                                <L 229>
    var_11 = norm_l2_0(var_9);
    // dif = end1 - end0                                                                      <L 230>
    var_12 = wp::sub(var_9, var_4);
    // dd = wp.dot(dif, dif)                                                                  <L 231>
    var_13 = wp::dot(var_12, var_12);
    // if (len0 <= radius) or (len1 <= radius) or (radius < MJ_MINVAL) or (len0 < MJ_MINVAL) or (len1 < MJ_MINVAL):       <L 234>
    var_15 = (var_10 <= var_radius);
    var_14 = var_15;
    if (!var_14) {
        var_16 = (var_11 <= var_radius);
        var_14 = var_14 || var_16;
    }
    if (!var_14) {
        var_18 = (var_radius < var_17);
        var_14 = var_14 || var_18;
    }
    if (!var_14) {
        var_19 = (var_10 < var_17);
        var_14 = var_14 || var_19;
    }
    if (!var_14) {
        var_20 = (var_11 < var_17);
        var_14 = var_14 || var_20;
    }
    if (var_14) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 235>
        var_23 = wp::vec_t<2, wp::float32>(var_22);
        var_24 = wp::vec_t<2, wp::float32>(var_22);
        ret_0 = var_21;
        ret_1 = var_23;
        ret_2 = var_24;
        return;
    }
    // if dd > MJ_MINVAL:                                                                     <L 238>
    var_25 = (var_13 > var_17);
    if (var_25) {
        // a = -wp.dot(dif, end0) / dd                                                        <L 240>
        var_26 = wp::dot(var_12, var_4);
        var_27 = wp::neg(var_26);
        var_28 = wp::div(var_27, var_13);
        // if (a > 0.0) and (a < 1.0):                                                        <L 243>
        var_31 = (var_28 > var_30);
        var_29 = var_31;
        if (var_29) {
            var_33 = (var_28 < var_32);
            var_29 = var_29 && var_33;
        }
        if (var_29) {
            // tmp = end0 + a * dif                                                           <L 244>
            var_34 = wp::mul(var_28, var_12);
            var_35 = wp::add(var_4, var_34);
            // if wp.norm_l2(tmp) <= radius:                                                  <L 245>
            var_36 = norm_l2_0(var_35);
            var_37 = (var_36 <= var_radius);
            if (var_37) {
                // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                        <L 246>
                var_39 = wp::vec_t<2, wp::float32>(var_22);
                var_40 = wp::vec_t<2, wp::float32>(var_22);
                ret_0 = var_38;
                ret_1 = var_39;
                ret_2 = var_40;
                return;
            }
        }
    }
    // pnt = 0.5 * (end0 + end1)                                                              <L 249>
    var_42 = wp::add(var_4, var_9);
    var_43 = wp::mul(var_41, var_42);
    // pnt, _ = math.normalize_with_norm(pnt)                                                 <L 250>
    normalize_with_norm_0(var_43, var_44, var_45);
    // pnt *= radius                                                                          <L 251>
    var_46 = wp::mul(var_44, var_radius);
    // A = math.safe_div(radius, len0)                                                        <L 254>
    var_47 = safe_div_0(var_radius, var_10);
    // B = math.safe_div(radius, len1)                                                        <L 255>
    var_48 = safe_div_0(var_radius, var_11);
    // sq_A = A * A                                                                           <L 256>
    var_49 = wp::mul(var_47, var_47);
    // sq_B = B * B                                                                           <L 257>
    var_50 = wp::mul(var_48, var_48);
    // cosG = math.safe_div(len0 * len0 + len1 * len1 - dd, 2.0 * len0 * len1)                <L 258>
    var_51 = wp::mul(var_10, var_10);
    var_52 = wp::mul(var_11, var_11);
    var_53 = wp::add(var_51, var_52);
    var_54 = wp::sub(var_53, var_13);
    var_56 = wp::mul(var_55, var_10);
    var_57 = wp::mul(var_56, var_11);
    var_58 = safe_div_0(var_54, var_57);
    // if cosG < -1.0 + MJ_MINVAL:                                                            <L 259>
    var_60 = wp::add(var_59, var_17);
    var_61 = (var_58 < var_60);
    if (var_61) {
        // return -1.0, pnt, pnt                                                              <L 260>
        ret_0 = var_62;
        ret_1 = var_46;
        ret_2 = var_46;
        return;
    }
    if (!var_61) {
        // elif cosG > 1.0 - MJ_MINVAL:                                                       <L 261>
        var_64 = wp::sub(var_63, var_17);
        var_65 = (var_58 > var_64);
        if (var_65) {
            // return 0.0, pnt, pnt                                                           <L 262>
            ret_0 = var_66;
            ret_1 = var_46;
            ret_2 = var_46;
            return;
        }
    }
    // G = wp.acos(cosG)                                                                      <L 263>
    var_67 = wp::acos(var_58);
    // z = zinit                                                                              <L 266>
    var_68 = wp::copy(var_zinit);
    // f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G                             <L 267>
    var_69 = wp::mul(var_47, var_68);
    var_70 = wp::asin(var_69);
    var_71 = wp::mul(var_48, var_68);
    var_72 = wp::asin(var_71);
    var_73 = wp::add(var_70, var_72);
    var_75 = wp::asin(var_68);
    var_76 = wp::mul(var_74, var_75);
    var_77 = wp::sub(var_73, var_76);
    var_78 = wp::add(var_77, var_67);
    // if f > 0.0:                                                                            <L 270>
    var_80 = (var_78 > var_79);
    if (var_80) {
        // return 0.0, pnt, pnt                                                               <L 271>
        ret_0 = var_81;
        ret_1 = var_46;
        ret_2 = var_46;
        return;
    }
    // iter = int(0)                                                                          <L 274>
    var_83 = wp::int(var_82);
    // while (iter < maxiter) and (wp.abs(f) > tolerance):                                    <L 276>
    start_while_5:;
    var_85 = (var_83 < var_maxiter);
    var_84 = var_85;
    if (var_84) {
        var_86 = wp::abs(var_78);
        var_87 = (var_86 > var_tolerance);
        var_84 = var_84 && var_87;
    }
    if ((var_84) == false) goto end_while_5;
        // sq_z = z * z                                                                       <L 278>
        var_88 = wp::mul(var_68, var_68);
        // df = (                                                                             <L 279>
        // A / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_A))                                  <L 280>
        var_90 = wp::mul(var_88, var_49);
        var_91 = wp::sub(var_89, var_90);
        var_92 = wp::sqrt(var_91);
        var_93 = wp::max(var_17, var_92);
        var_94 = wp::div(var_47, var_93);
        // + B / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_B))                                <L 281>
        var_96 = wp::mul(var_88, var_50);
        var_97 = wp::sub(var_95, var_96);
        var_98 = wp::sqrt(var_97);
        var_99 = wp::max(var_17, var_98);
        var_100 = wp::div(var_48, var_99);
        var_101 = wp::add(var_94, var_100);
        // - 2.0 / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z))                                     <L 282>
        var_104 = wp::sub(var_103, var_88);
        var_105 = wp::sqrt(var_104);
        var_106 = wp::max(var_17, var_105);
        var_107 = wp::div(var_102, var_106);
        var_108 = wp::sub(var_101, var_107);
        // if df > -MJ_MINVAL:                                                                <L 286>
        var_110 = (var_108 > var_109);
        if (var_110) {
            // return 0.0, pnt, pnt                                                           <L 287>
            ret_0 = var_111;
            ret_1 = var_46;
            ret_2 = var_46;
            return;
        }
        // z1 = z - math.safe_div(f, df)                                                      <L 290>
        var_112 = safe_div_0(var_78, var_108);
        var_113 = wp::sub(var_68, var_112);
        // if z1 > z:                                                                         <L 293>
        var_114 = (var_113 > var_68);
        if (var_114) {
            // return 0.0, pnt, pnt                                                           <L 294>
            ret_0 = var_115;
            ret_1 = var_46;
            ret_2 = var_46;
            return;
        }
        // z = z1                                                                             <L 297>
        var_116 = wp::copy(var_113);
        // f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G                         <L 298>
        var_117 = wp::mul(var_47, var_116);
        var_118 = wp::asin(var_117);
        var_119 = wp::mul(var_48, var_116);
        var_120 = wp::asin(var_119);
        var_121 = wp::add(var_118, var_120);
        var_123 = wp::asin(var_116);
        var_124 = wp::mul(var_122, var_123);
        var_125 = wp::sub(var_121, var_124);
        var_126 = wp::add(var_125, var_67);
        // if f > tolerance:                                                                  <L 301>
        var_127 = (var_126 > var_tolerance);
        if (var_127) {
            // return 0.0, pnt, pnt                                                           <L 302>
            ret_0 = var_128;
            ret_1 = var_46;
            ret_2 = var_46;
            return;
        }
        // iter += 1                                                                          <L 304>
        var_130 = wp::add(var_83, var_129);
        wp::assign(var_68, var_116);
        wp::assign(var_78, var_126);
        wp::assign(var_83, var_130);
    goto start_while_5;
    end_while_5:;
    // if iter >= maxiter:                                                                    <L 307>
    var_131 = (var_83 >= var_maxiter);
    if (var_131) {
        // return 0.0, pnt, pnt                                                               <L 308>
        ret_0 = var_132;
        ret_1 = var_46;
        ret_2 = var_46;
        return;
    }
    // if end[0] * end[3] - end[1] * end[2] > 0.0:                                            <L 311>
    var_134 = wp::extract(var_end, var_133);
    var_136 = wp::extract(var_end, var_135);
    var_137 = wp::mul(var_134, var_136);
    var_139 = wp::extract(var_end, var_138);
    var_141 = wp::extract(var_end, var_140);
    var_142 = wp::mul(var_139, var_141);
    var_143 = wp::sub(var_137, var_142);
    var_145 = (var_143 > var_144);
    if (var_145) {
        // vec = end0                                                                         <L 312>
        var_146 = wp::copy(var_4);
        // ang = wp.asin(z) - wp.asin(A * z)                                                  <L 313>
        var_147 = wp::asin(var_68);
        var_148 = wp::mul(var_47, var_68);
        var_149 = wp::asin(var_148);
        var_150 = wp::sub(var_147, var_149);
    }
    if (!var_145) {
        // vec = end1                                                                         <L 315>
        var_151 = wp::copy(var_9);
        // ang = wp.asin(z) - wp.asin(B * z)                                                  <L 316>
        var_152 = wp::asin(var_68);
        var_153 = wp::mul(var_48, var_68);
        var_154 = wp::asin(var_153);
        var_155 = wp::sub(var_152, var_154);
    }
    var_156 = wp::where(var_145, var_146, var_151);
    var_157 = wp::where(var_145, var_150, var_155);
    // vec, _ = math.normalize_with_norm(vec)                                                 <L 318>
    normalize_with_norm_0(var_156, var_158, var_159);
    // pnt = wp.vec2(                                                                         <L 319>
    // radius * (wp.cos(ang) * vec[0] - wp.sin(ang) * vec[1]),                                <L 320>
    var_160 = wp::cos(var_157);
    var_162 = wp::extract(var_158, var_161);
    var_163 = wp::mul(var_160, var_162);
    var_164 = wp::sin(var_157);
    var_166 = wp::extract(var_158, var_165);
    var_167 = wp::mul(var_164, var_166);
    var_168 = wp::sub(var_163, var_167);
    var_169 = wp::mul(var_radius, var_168);
    // radius * (wp.sin(ang) * vec[0] + wp.cos(ang) * vec[1]),                                <L 321>
    var_170 = wp::sin(var_157);
    var_172 = wp::extract(var_158, var_171);
    var_173 = wp::mul(var_170, var_172);
    var_174 = wp::cos(var_157);
    var_176 = wp::extract(var_158, var_175);
    var_177 = wp::mul(var_174, var_176);
    var_178 = wp::add(var_173, var_177);
    var_179 = wp::mul(var_radius, var_178);
    var_180 = wp::vec_t<2, wp::float32>(var_169, var_179);
    // return 0.0, pnt, pnt                                                                   <L 324>
    ret_0 = var_181;
    ret_1 = var_180;
    ret_2 = var_180;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:31
static CUDA_CALLABLE bool is_intersect_0(
    wp::vec_t<2, wp::float32> var_p1,
    wp::vec_t<2, wp::float32> var_p2,
    wp::vec_t<2, wp::float32> var_p3,
    wp::vec_t<2, wp::float32> var_p4)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1e-15;
    bool var_25;
    const bool var_26 = false;
    const wp::int32 var_27 = 0;
    wp::float32 var_28;
    const wp::int32 var_29 = 0;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    bool var_75;
    const wp::int32 var_76 = 0;
    bool var_77;
    const wp::float32 var_78 = 1.0;
    bool var_79;
    const wp::float32 var_80 = 0.0;
    bool var_81;
    const wp::float32 var_82 = 1.0;
    bool var_83;
    const bool var_84 = true;
    const bool var_85 = false;
    //---------
    // forward
    // def is_intersect(p1: wp.vec2, p2: wp.vec2, p3: wp.vec2, p4: wp.vec2) -> bool:          <L 32>
    // det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])            <L 45>
    var_1 = wp::extract(var_p4, var_0);
    var_3 = wp::extract(var_p3, var_2);
    var_4 = wp::sub(var_1, var_3);
    var_6 = wp::extract(var_p2, var_5);
    var_8 = wp::extract(var_p1, var_7);
    var_9 = wp::sub(var_6, var_8);
    var_10 = wp::mul(var_4, var_9);
    var_12 = wp::extract(var_p4, var_11);
    var_14 = wp::extract(var_p3, var_13);
    var_15 = wp::sub(var_12, var_14);
    var_17 = wp::extract(var_p2, var_16);
    var_19 = wp::extract(var_p1, var_18);
    var_20 = wp::sub(var_17, var_19);
    var_21 = wp::mul(var_15, var_20);
    var_22 = wp::sub(var_10, var_21);
    // if wp.abs(det) < MJ_MINVAL:                                                            <L 47>
    var_23 = wp::abs(var_22);
    var_25 = (var_23 < var_24);
    if (var_25) {
        // return False                                                                       <L 48>
        return var_26;
    }
    // a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / det       <L 51>
    var_28 = wp::extract(var_p4, var_27);
    var_30 = wp::extract(var_p3, var_29);
    var_31 = wp::sub(var_28, var_30);
    var_33 = wp::extract(var_p1, var_32);
    var_35 = wp::extract(var_p3, var_34);
    var_36 = wp::sub(var_33, var_35);
    var_37 = wp::mul(var_31, var_36);
    var_39 = wp::extract(var_p4, var_38);
    var_41 = wp::extract(var_p3, var_40);
    var_42 = wp::sub(var_39, var_41);
    var_44 = wp::extract(var_p1, var_43);
    var_46 = wp::extract(var_p3, var_45);
    var_47 = wp::sub(var_44, var_46);
    var_48 = wp::mul(var_42, var_47);
    var_49 = wp::sub(var_37, var_48);
    var_50 = wp::div(var_49, var_22);
    // b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / det       <L 52>
    var_52 = wp::extract(var_p2, var_51);
    var_54 = wp::extract(var_p1, var_53);
    var_55 = wp::sub(var_52, var_54);
    var_57 = wp::extract(var_p1, var_56);
    var_59 = wp::extract(var_p3, var_58);
    var_60 = wp::sub(var_57, var_59);
    var_61 = wp::mul(var_55, var_60);
    var_63 = wp::extract(var_p2, var_62);
    var_65 = wp::extract(var_p1, var_64);
    var_66 = wp::sub(var_63, var_65);
    var_68 = wp::extract(var_p1, var_67);
    var_70 = wp::extract(var_p3, var_69);
    var_71 = wp::sub(var_68, var_70);
    var_72 = wp::mul(var_66, var_71);
    var_73 = wp::sub(var_61, var_72);
    var_74 = wp::div(var_73, var_22);
    // if a >= 0 and a <= 1.0 and b >= 0.0 and b <= 1.0:                                      <L 54>
    var_77 = (var_50 >= var_76);
    var_75 = var_77;
    if (var_75) {
        var_79 = (var_50 <= var_78);
        var_75 = var_75 && var_79;
    }
    if (var_75) {
        var_81 = (var_74 >= var_80);
        var_75 = var_75 && var_81;
    }
    if (var_75) {
        var_83 = (var_74 <= var_82);
        var_75 = var_75 && var_83;
    }
    if (var_75) {
        // return True                                                                        <L 55>
        return var_84;
    }
    if (!var_75) {
        // return False                                                                       <L 57>
        return var_85;
    }
    return {};
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:77
static CUDA_CALLABLE wp::float32 length_circle_0(
    wp::vec_t<2, wp::float32> var_p0,
    wp::vec_t<2, wp::float32> var_p1,
    wp::int32 var_ind,
    wp::float32 var_radius)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::float32> var_0;
    wp::float32 var_1;
    wp::vec_t<2, wp::float32> var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    bool var_17;
    bool var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    const wp::int32 var_21 = 0;
    bool var_22;
    bool var_23;
    const wp::float32 var_24 = 0.0;
    bool var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    const wp::float32 var_28 = 2.0;
    const wp::float32 var_29 = 3.141592653589793;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    //---------
    // forward
    // def length_circle(p0: wp.vec2, p1: wp.vec2, ind: int, radius: float) -> float:         <L 78>
    // p0n, _ = math.normalize_with_norm(p0)                                                  <L 91>
    normalize_with_norm_0(var_p0, var_0, var_1);
    // p1n, _ = math.normalize_with_norm(p1)                                                  <L 92>
    normalize_with_norm_0(var_p1, var_2, var_3);
    // angle = wp.acos(wp.dot(p0n, p1n))                                                      <L 94>
    var_4 = wp::dot(var_0, var_2);
    var_5 = wp::acos(var_4);
    // cross = p0[1] * p1[0] - p0[0] * p1[1]                                                  <L 97>
    var_7 = wp::extract(var_p0, var_6);
    var_9 = wp::extract(var_p1, var_8);
    var_10 = wp::mul(var_7, var_9);
    var_12 = wp::extract(var_p0, var_11);
    var_14 = wp::extract(var_p1, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    // if (cross > 0.0 and ind != 0) or (cross < 0.0 and ind == 0):                           <L 98>
    var_20 = (var_16 > var_19);
    var_18 = var_20;
    if (var_18) {
        var_22 = (var_ind != var_21);
        var_18 = var_18 && var_22;
    }
    var_17 = var_18;
    if (!var_17) {
        var_25 = (var_16 < var_24);
        var_23 = var_25;
        if (var_23) {
            var_27 = (var_ind == var_26);
            var_23 = var_23 && var_27;
        }
        var_17 = var_17 || var_23;
    }
    if (var_17) {
        // angle = 2.0 * wp.pi - angle                                                        <L 99>
        var_30 = wp::mul(var_28, var_29);
        var_31 = wp::sub(var_30, var_5);
    }
    var_32 = wp::where(var_17, var_31, var_5);
    // return radius * angle                                                                  <L 101>
    var_33 = wp::mul(var_radius, var_32);
    return var_33;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:104
static CUDA_CALLABLE void wrap_circle_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::vec_t<2, wp::float32> var_side,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 10000000000.0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    wp::vec_t<2, wp::float32> var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::int32 var_10 = 3;
    wp::float32 var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    bool var_16;
    bool var_17;
    bool var_18;
    const wp::float32 var_19 = 1e-15;
    bool var_20;
    const wp::float32 var_21 = -1.0;
    wp::vec_t<2, wp::float32> var_22;
    wp::vec_t<2, wp::float32> var_23;
    wp::vec_t<2, wp::float32> var_24;
    wp::float32 var_25;
    bool var_26;
    const wp::float32 var_27 = -1.0;
    wp::vec_t<2, wp::float32> var_28;
    wp::vec_t<2, wp::float32> var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 1.0;
    wp::float32 var_35;
    wp::vec_t<2, wp::float32> var_36;
    wp::vec_t<2, wp::float32> var_37;
    bool var_38;
    wp::float32 var_39;
    bool var_40;
    bool var_41;
    bool var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 0.0;
    bool var_45;
    const wp::float32 var_46 = -1.0;
    wp::vec_t<2, wp::float32> var_47;
    wp::vec_t<2, wp::float32> var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::int32 var_65 = 0;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::vec_t<2, wp::float32> var_71;
    const wp::int32 var_72 = 2;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 3;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 3;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::int32 var_84 = 2;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::vec_t<2, wp::float32> var_90;
    const wp::int32 var_91 = 0;
    wp::float32 var_92;
    wp::float32 var_93;
    const wp::int32 var_94 = 1;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    const wp::int32 var_100 = 1;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 0;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::vec_t<2, wp::float32> var_109;
    const wp::int32 var_110 = 2;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 3;
    wp::float32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    const wp::int32 var_119 = 3;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 2;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::vec_t<2, wp::float32> var_128;
    wp::vec_t<2, wp::float32> var_129;
    wp::vec_t<2, wp::float32> var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::vec_t<2, wp::float32> var_133;
    wp::vec_t<2, wp::float32> var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::vec_t<2, wp::float32> var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::vec_t<2, wp::float32> var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::vec_t<2, wp::float32> var_143;
    wp::float32 var_144;
    wp::vec_t<2, wp::float32> var_145;
    wp::float32 var_146;
    bool var_147;
    const wp::float32 var_148 = -10000.0;
    wp::float32 var_149;
    bool var_150;
    const wp::float32 var_151 = -10000.0;
    wp::float32 var_152;
    bool var_153;
    wp::vec_t<2, wp::float32> var_154;
    wp::vec_t<2, wp::float32> var_155;
    const wp::int32 var_156 = 0;
    wp::vec_t<2, wp::float32> var_157;
    wp::vec_t<2, wp::float32> var_158;
    const wp::int32 var_159 = 1;
    wp::vec_t<2, wp::float32> var_160;
    wp::vec_t<2, wp::float32> var_161;
    wp::int32 var_162;
    bool var_163;
    const wp::float32 var_164 = -1.0;
    wp::vec_t<2, wp::float32> var_165;
    wp::vec_t<2, wp::float32> var_166;
    wp::float32 var_167;
    //---------
    // forward
    // def wrap_circle(end: wp.vec4, side: wp.vec2, radius: float) -> Tuple[float, wp.vec2, wp.vec2]:       <L 105>
    // valid_side = wp.norm_l2(side) < MJ_MAXVAL                                              <L 116>
    var_0 = norm_l2_0(var_side);
    var_2 = (var_0 < var_1);
    // end0 = wp.vec2(end[0], end[1])                                                         <L 118>
    var_4 = wp::extract(var_end, var_3);
    var_6 = wp::extract(var_end, var_5);
    var_7 = wp::vec_t<2, wp::float32>(var_4, var_6);
    // end1 = wp.vec2(end[2], end[3])                                                         <L 119>
    var_9 = wp::extract(var_end, var_8);
    var_11 = wp::extract(var_end, var_10);
    var_12 = wp::vec_t<2, wp::float32>(var_9, var_11);
    // sqlen0 = wp.dot(end0, end0)                                                            <L 121>
    var_13 = wp::dot(var_7, var_7);
    // sqlen1 = wp.dot(end1, end1)                                                            <L 122>
    var_14 = wp::dot(var_12, var_12);
    // sqrad = radius * radius                                                                <L 123>
    var_15 = wp::mul(var_radius, var_radius);
    // if (sqlen0 < sqrad) or (sqlen1 < sqrad) or (radius < MJ_MINVAL):                       <L 126>
    var_17 = (var_13 < var_15);
    var_16 = var_17;
    if (!var_16) {
        var_18 = (var_14 < var_15);
        var_16 = var_16 || var_18;
    }
    if (!var_16) {
        var_20 = (var_radius < var_19);
        var_16 = var_16 || var_20;
    }
    if (var_16) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 127>
        var_22 = wp::vec_t<2, wp::float32>(var_1);
        var_23 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_21;
        ret_1 = var_22;
        ret_2 = var_23;
        return;
    }
    // dif = end1 - end0                                                                      <L 130>
    var_24 = wp::sub(var_12, var_7);
    // dd = wp.dot(dif, dif)                                                                  <L 131>
    var_25 = wp::dot(var_24, var_24);
    // if dd < MJ_MINVAL:                                                                     <L 132>
    var_26 = (var_25 < var_19);
    if (var_26) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 133>
        var_28 = wp::vec_t<2, wp::float32>(var_1);
        var_29 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_27;
        ret_1 = var_28;
        ret_2 = var_29;
        return;
    }
    // a = -wp.dot(dif, end0) / dd                                                            <L 136>
    var_30 = wp::dot(var_24, var_7);
    var_31 = wp::neg(var_30);
    var_32 = wp::div(var_31, var_25);
    // a = wp.clamp(a, 0.0, 1.0)                                                              <L 137>
    var_35 = wp::clamp(var_32, var_33, var_34);
    // tmp = a * dif + end0                                                                   <L 140>
    var_36 = wp::mul(var_35, var_24);
    var_37 = wp::add(var_36, var_7);
    // if (wp.dot(tmp, tmp) > sqrad) and (not valid_side or wp.dot(side, tmp) >= 0.0):        <L 141>
    var_39 = wp::dot(var_37, var_37);
    var_40 = (var_39 > var_15);
    var_38 = var_40;
    if (var_38) {
        var_42 = wp::unot(var_2);
        var_41 = var_42;
        if (!var_41) {
            var_43 = wp::dot(var_side, var_37);
            var_45 = (var_43 >= var_44);
            var_41 = var_41 || var_45;
        }
        var_38 = var_38 && var_41;
    }
    if (var_38) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 142>
        var_47 = wp::vec_t<2, wp::float32>(var_1);
        var_48 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_46;
        ret_1 = var_47;
        ret_2 = var_48;
        return;
    }
    // sqrt0 = wp.sqrt(sqlen0 - sqrad)                                                        <L 144>
    var_49 = wp::sub(var_13, var_15);
    var_50 = wp::sqrt(var_49);
    // sqrt1 = wp.sqrt(sqlen1 - sqrad)                                                        <L 145>
    var_51 = wp::sub(var_14, var_15);
    var_52 = wp::sqrt(var_51);
    // sol00 = wp.vec2(                                                                       <L 148>
    // math.safe_div(end[0] * sqrad + radius * end[1] * sqrt0, sqlen0),                       <L 149>
    var_54 = wp::extract(var_end, var_53);
    var_55 = wp::mul(var_54, var_15);
    var_57 = wp::extract(var_end, var_56);
    var_58 = wp::mul(var_radius, var_57);
    var_59 = wp::mul(var_58, var_50);
    var_60 = wp::add(var_55, var_59);
    var_61 = safe_div_0(var_60, var_13);
    // math.safe_div(end[1] * sqrad - radius * end[0] * sqrt0, sqlen0),                       <L 150>
    var_63 = wp::extract(var_end, var_62);
    var_64 = wp::mul(var_63, var_15);
    var_66 = wp::extract(var_end, var_65);
    var_67 = wp::mul(var_radius, var_66);
    var_68 = wp::mul(var_67, var_50);
    var_69 = wp::sub(var_64, var_68);
    var_70 = safe_div_0(var_69, var_13);
    var_71 = wp::vec_t<2, wp::float32>(var_61, var_70);
    // sol01 = wp.vec2(                                                                       <L 152>
    // math.safe_div(end[2] * sqrad - radius * end[3] * sqrt1, sqlen1),                       <L 153>
    var_73 = wp::extract(var_end, var_72);
    var_74 = wp::mul(var_73, var_15);
    var_76 = wp::extract(var_end, var_75);
    var_77 = wp::mul(var_radius, var_76);
    var_78 = wp::mul(var_77, var_52);
    var_79 = wp::sub(var_74, var_78);
    var_80 = safe_div_0(var_79, var_14);
    // math.safe_div(end[3] * sqrad + radius * end[2] * sqrt1, sqlen1),                       <L 154>
    var_82 = wp::extract(var_end, var_81);
    var_83 = wp::mul(var_82, var_15);
    var_85 = wp::extract(var_end, var_84);
    var_86 = wp::mul(var_radius, var_85);
    var_87 = wp::mul(var_86, var_52);
    var_88 = wp::add(var_83, var_87);
    var_89 = safe_div_0(var_88, var_14);
    var_90 = wp::vec_t<2, wp::float32>(var_80, var_89);
    // sol10 = wp.vec2(                                                                       <L 157>
    // math.safe_div(end[0] * sqrad - radius * end[1] * sqrt0, sqlen0),                       <L 158>
    var_92 = wp::extract(var_end, var_91);
    var_93 = wp::mul(var_92, var_15);
    var_95 = wp::extract(var_end, var_94);
    var_96 = wp::mul(var_radius, var_95);
    var_97 = wp::mul(var_96, var_50);
    var_98 = wp::sub(var_93, var_97);
    var_99 = safe_div_0(var_98, var_13);
    // math.safe_div(end[1] * sqrad + radius * end[0] * sqrt0, sqlen0),                       <L 159>
    var_101 = wp::extract(var_end, var_100);
    var_102 = wp::mul(var_101, var_15);
    var_104 = wp::extract(var_end, var_103);
    var_105 = wp::mul(var_radius, var_104);
    var_106 = wp::mul(var_105, var_50);
    var_107 = wp::add(var_102, var_106);
    var_108 = safe_div_0(var_107, var_13);
    var_109 = wp::vec_t<2, wp::float32>(var_99, var_108);
    // sol11 = wp.vec2(                                                                       <L 161>
    // math.safe_div(end[2] * sqrad + radius * end[3] * sqrt1, sqlen1),                       <L 162>
    var_111 = wp::extract(var_end, var_110);
    var_112 = wp::mul(var_111, var_15);
    var_114 = wp::extract(var_end, var_113);
    var_115 = wp::mul(var_radius, var_114);
    var_116 = wp::mul(var_115, var_52);
    var_117 = wp::add(var_112, var_116);
    var_118 = safe_div_0(var_117, var_14);
    // math.safe_div(end[3] * sqrad - radius * end[2] * sqrt1, sqlen1),                       <L 163>
    var_120 = wp::extract(var_end, var_119);
    var_121 = wp::mul(var_120, var_15);
    var_123 = wp::extract(var_end, var_122);
    var_124 = wp::mul(var_radius, var_123);
    var_125 = wp::mul(var_124, var_52);
    var_126 = wp::sub(var_121, var_125);
    var_127 = safe_div_0(var_126, var_14);
    var_128 = wp::vec_t<2, wp::float32>(var_118, var_127);
    // if valid_side:                                                                         <L 167>
    if (var_2) {
        // tmp0, _ = math.normalize_with_norm(sol00 + sol01)                                  <L 168>
        var_129 = wp::add(var_71, var_90);
        normalize_with_norm_0(var_129, var_130, var_131);
        // good0 = wp.dot(tmp0, side)                                                         <L 169>
        var_132 = wp::dot(var_130, var_side);
        // tmp1, _ = math.normalize_with_norm(sol10 + sol11)                                  <L 170>
        var_133 = wp::add(var_109, var_128);
        normalize_with_norm_0(var_133, var_134, var_135);
        // good1 = wp.dot(tmp1, side)                                                         <L 171>
        var_136 = wp::dot(var_134, var_side);
    }
    if (!var_2) {
        // tmp0 = sol00 - sol01                                                               <L 173>
        var_137 = wp::sub(var_71, var_90);
        // good0 = -wp.dot(tmp0, tmp0)                                                        <L 174>
        var_138 = wp::dot(var_137, var_137);
        var_139 = wp::neg(var_138);
        // tmp1 = sol10 - sol11                                                               <L 175>
        var_140 = wp::sub(var_109, var_128);
        // good1 = -wp.dot(tmp1, tmp1)                                                        <L 176>
        var_141 = wp::dot(var_140, var_140);
        var_142 = wp::neg(var_141);
    }
    var_143 = wp::where(var_2, var_130, var_137);
    var_144 = wp::where(var_2, var_132, var_139);
    var_145 = wp::where(var_2, var_134, var_140);
    var_146 = wp::where(var_2, var_136, var_142);
    // if is_intersect(end0, sol00, end1, sol01):                                             <L 179>
    var_147 = is_intersect_0(var_7, var_71, var_12, var_90);
    if (var_147) {
        // good0 = -10000.0                                                                   <L 180>
    }
    var_149 = wp::where(var_147, var_148, var_144);
    // if is_intersect(end0, sol10, end1, sol11):                                             <L 181>
    var_150 = is_intersect_0(var_7, var_109, var_12, var_128);
    if (var_150) {
        // good1 = -10000.0                                                                   <L 182>
    }
    var_152 = wp::where(var_150, var_151, var_146);
    // if good0 > good1:                                                                      <L 185>
    var_153 = (var_149 > var_152);
    if (var_153) {
        // pnt0 = sol00                                                                       <L 186>
        var_154 = wp::copy(var_71);
        // pnt1 = sol01                                                                       <L 187>
        var_155 = wp::copy(var_90);
        // ind = 0                                                                            <L 188>
    }
    if (!var_153) {
        // pnt0 = sol10                                                                       <L 190>
        var_157 = wp::copy(var_109);
        // pnt1 = sol11                                                                       <L 191>
        var_158 = wp::copy(var_128);
        // ind = 1                                                                            <L 192>
    }
    var_160 = wp::where(var_153, var_154, var_157);
    var_161 = wp::where(var_153, var_155, var_158);
    var_162 = wp::where(var_153, var_156, var_159);
    // if is_intersect(end0, pnt0, end1, pnt1):                                               <L 195>
    var_163 = is_intersect_0(var_7, var_160, var_12, var_161);
    if (var_163) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 196>
        var_165 = wp::vec_t<2, wp::float32>(var_1);
        var_166 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_164;
        ret_1 = var_165;
        ret_2 = var_166;
        return;
    }
    // return length_circle(pnt0, pnt1, ind, radius), pnt0, pnt1                              <L 199>
    var_167 = length_circle_0(var_160, var_161, var_162, var_radius);
    ret_0 = var_167;
    ret_1 = var_160;
    ret_2 = var_161;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:327
static CUDA_CALLABLE void wrap_0(
    wp::vec_t<3, wp::float32> var_x0,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::float32 var_radius,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_side,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 4;
    bool var_2;
    const wp::int32 var_3 = 5;
    bool var_4;
    const wp::float32 var_5 = 10000000000.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    bool var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 1e-15;
    bool var_16;
    wp::float32 var_17;
    bool var_18;
    const wp::float32 var_19 = -1.0;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    const wp::int32 var_22 = 4;
    bool var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    bool var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    wp::int32 var_32;
    bool var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    bool var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    const wp::int32 var_41 = 2;
    wp::float32 var_42;
    bool var_43;
    const wp::int32 var_44 = 1;
    wp::int32 var_45;
    bool var_46;
    const wp::int32 var_47 = 2;
    wp::float32 var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    bool var_51;
    const wp::int32 var_52 = 2;
    wp::float32 var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    bool var_56;
    const wp::int32 var_57 = 2;
    wp::int32 var_58;
    const wp::float32 var_59 = 1.0;
    wp::vec_t<3, wp::float32> var_60;
    const wp::float32 var_61 = 0.0;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::float32 var_69;
    const wp::float32 var_70 = 1.0;
    const wp::float32 var_71 = 0.0;
    const wp::float32 var_72 = 0.0;
    wp::vec_t<3, wp::float32> var_73;
    const wp::float32 var_74 = 0.0;
    const wp::float32 var_75 = 1.0;
    const wp::float32 var_76 = 0.0;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::vec_t<4, wp::float32> var_84;
    wp::float32 var_85;
    bool var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::vec_t<2, wp::float32> var_91;
    wp::vec_t<2, wp::float32> var_92;
    wp::float32 var_93;
    wp::vec_t<2, wp::float32> var_94;
    wp::vec_t<2, wp::float32> var_95;
    wp::float32 var_96;
    wp::vec_t<2, wp::float32> var_97;
    bool var_98;
    wp::float32 var_99;
    bool var_100;
    wp::float32 var_101;
    wp::vec_t<2, wp::float32> var_102;
    wp::vec_t<2, wp::float32> var_103;
    const wp::int32 var_104 = 20;
    const wp::float32 var_105 = 0.9999999;
    const wp::float32 var_106 = 1e-06;
    wp::float32 var_107;
    wp::vec_t<2, wp::float32> var_108;
    wp::vec_t<2, wp::float32> var_109;
    wp::float32 var_110;
    wp::vec_t<2, wp::float32> var_111;
    wp::vec_t<2, wp::float32> var_112;
    const wp::float32 var_113 = 0.0;
    bool var_114;
    const wp::float32 var_115 = -1.0;
    wp::vec_t<3, wp::float32> var_116;
    wp::vec_t<3, wp::float32> var_117;
    const wp::int32 var_118 = 0;
    wp::float32 var_119;
    wp::vec_t<3, wp::float32> var_120;
    const wp::int32 var_121 = 1;
    wp::float32 var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::vec_t<3, wp::float32> var_124;
    const wp::int32 var_125 = 0;
    wp::float32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    const wp::int32 var_132 = 5;
    bool var_133;
    const wp::int32 var_134 = 0;
    wp::float32 var_135;
    const wp::int32 var_136 = 0;
    wp::float32 var_137;
    wp::float32 var_138;
    const wp::int32 var_139 = 0;
    wp::float32 var_140;
    const wp::int32 var_141 = 0;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    const wp::int32 var_145 = 1;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    wp::float32 var_149;
    const wp::int32 var_150 = 1;
    wp::float32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    const wp::int32 var_158 = 0;
    wp::float32 var_159;
    const wp::int32 var_160 = 0;
    wp::float32 var_161;
    wp::float32 var_162;
    const wp::int32 var_163 = 0;
    wp::float32 var_164;
    const wp::int32 var_165 = 0;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    const wp::int32 var_169 = 1;
    wp::float32 var_170;
    const wp::int32 var_171 = 1;
    wp::float32 var_172;
    wp::float32 var_173;
    const wp::int32 var_174 = 1;
    wp::float32 var_175;
    const wp::int32 var_176 = 1;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    wp::float32 var_180;
    wp::float32 var_181;
    const wp::int32 var_182 = 2;
    wp::float32 var_183;
    const wp::int32 var_184 = 2;
    wp::float32 var_185;
    const wp::int32 var_186 = 2;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::float32 var_191;
    wp::float32 var_192;
    wp::float32 var_193;
    const wp::int32 var_194 = 2;
    const wp::int32 var_195 = 2;
    wp::float32 var_196;
    const wp::int32 var_197 = 2;
    wp::float32 var_198;
    const wp::int32 var_199 = 2;
    wp::float32 var_200;
    wp::float32 var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    wp::float32 var_205;
    wp::float32 var_206;
    wp::float32 var_207;
    const wp::int32 var_208 = 2;
    const wp::int32 var_209 = 2;
    wp::float32 var_210;
    const wp::int32 var_211 = 2;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::float32 var_214;
    wp::float32 var_215;
    wp::float32 var_216;
    wp::float32 var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::vec_t<3, wp::float32> var_220;
    wp::vec_t<3, wp::float32> var_221;
    wp::vec_t<3, wp::float32> var_222;
    wp::vec_t<3, wp::float32> var_223;
    //---------
    // forward
    // def wrap(                                                                              <L 328>
    // if geomtype != WrapType.SPHERE and geomtype != WrapType.CYLINDER:                      <L 346>
    var_2 = (var_geomtype != var_1);
    var_0 = var_2;
    if (var_0) {
        var_4 = (var_geomtype != var_3);
        var_0 = var_0 && var_4;
    }
    if (var_0) {
        // return MJ_MAXVAL, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                           <L 347>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        var_7 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_5;
        ret_1 = var_6;
        ret_2 = var_7;
        return;
    }
    // matT = wp.transpose(mat)                                                               <L 350>
    var_8 = wp::transpose(var_mat);
    // p0 = matT @ (x0 - pos)                                                                 <L 351>
    var_9 = wp::sub(var_x0, var_pos);
    var_10 = wp::mul(var_8, var_9);
    // p1 = matT @ (x1 - pos)                                                                 <L 352>
    var_11 = wp::sub(var_x1, var_pos);
    var_12 = wp::mul(var_8, var_11);
    // if (wp.norm_l2(p0) < MJ_MINVAL) or (wp.norm_l2(p1) < MJ_MINVAL):                       <L 355>
    var_14 = norm_l2_0(var_10);
    var_16 = (var_14 < var_15);
    var_13 = var_16;
    if (!var_13) {
        var_17 = norm_l2_0(var_12);
        var_18 = (var_17 < var_15);
        var_13 = var_13 || var_18;
    }
    if (var_13) {
        // return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                                <L 356>
        var_20 = wp::vec_t<3, wp::float32>(var_5);
        var_21 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_19;
        ret_1 = var_20;
        ret_2 = var_21;
        return;
    }
    // if geomtype == WrapType.SPHERE:                                                        <L 359>
    var_23 = (var_geomtype == var_22);
    if (var_23) {
        // axis0, _ = math.normalize_with_norm(p0)                                            <L 361>
        normalize_with_norm_0(var_10, var_24, var_25);
        // normal = wp.cross(p0, p1)                                                          <L 364>
        var_26 = wp::cross(var_10, var_12);
        // normal, nrm = math.normalize_with_norm(normal)                                     <L 365>
        normalize_with_norm_0(var_26, var_27, var_28);
        // if nrm < MJ_MINVAL:                                                                <L 368>
        var_29 = (var_28 < var_15);
        if (var_29) {
            // axis0_abs = wp.abs(axis0)                                                      <L 370>
            var_30 = wp::abs(var_24);
            // i = int(0)                                                                     <L 371>
            var_32 = wp::int(var_31);
            // if (axis0_abs[1] > axis0_abs[0]) and (axis0_abs[1] > axis0_abs[2]):            <L 372>
            var_35 = wp::extract(var_30, var_34);
            var_37 = wp::extract(var_30, var_36);
            var_38 = (var_35 > var_37);
            var_33 = var_38;
            if (var_33) {
                var_40 = wp::extract(var_30, var_39);
                var_42 = wp::extract(var_30, var_41);
                var_43 = (var_40 > var_42);
                var_33 = var_33 && var_43;
            }
            if (var_33) {
                // i = 1                                                                      <L 373>
            }
            var_45 = wp::where(var_33, var_44, var_32);
            // if (axis0_abs[2] > axis0_abs[0]) and (axis0_abs[2] > axis0_abs[1]):            <L 374>
            var_48 = wp::extract(var_30, var_47);
            var_50 = wp::extract(var_30, var_49);
            var_51 = (var_48 > var_50);
            var_46 = var_51;
            if (var_46) {
                var_53 = wp::extract(var_30, var_52);
                var_55 = wp::extract(var_30, var_54);
                var_56 = (var_53 > var_55);
                var_46 = var_46 && var_56;
            }
            if (var_46) {
                // i = 2                                                                      <L 375>
            }
            var_58 = wp::where(var_46, var_57, var_45);
            // axis1 = wp.vec3(1.0)                                                           <L 378>
            var_60 = wp::vec_t<3, wp::float32>(var_59);
            // axis1[i] = 0.0                                                                 <L 379>
            wp::assign_inplace(var_60, var_58, var_61);
            // normal = wp.cross(axis0, axis1)                                                <L 382>
            var_62 = wp::cross(var_24, var_60);
            // normal, _ = math.normalize_with_norm(normal)                                   <L 383>
            normalize_with_norm_0(var_62, var_63, var_64);
        }
        var_65 = wp::where(var_29, var_64, var_25);
        var_66 = wp::where(var_29, var_63, var_27);
        // axis1 = wp.cross(normal, axis0)                                                    <L 386>
        var_67 = wp::cross(var_66, var_24);
        // axis1, _ = math.normalize_with_norm(axis1)                                         <L 387>
        normalize_with_norm_0(var_67, var_68, var_69);
    }
    if (!var_23) {
        // axis0 = wp.vec3(1.0, 0.0, 0.0)                                                     <L 390>
        var_73 = wp::vec_t<3, wp::float32>(var_70, var_71, var_72);
        // axis1 = wp.vec3(0.0, 1.0, 0.0)                                                     <L 393>
        var_77 = wp::vec_t<3, wp::float32>(var_74, var_75, var_76);
    }
    var_78 = wp::where(var_23, var_24, var_73);
    var_79 = wp::where(var_23, var_68, var_77);
    // end = wp.vec4(                                                                         <L 396>
    // wp.dot(p0, axis0),                                                                     <L 397>
    var_80 = wp::dot(var_10, var_78);
    // wp.dot(p0, axis1),                                                                     <L 398>
    var_81 = wp::dot(var_10, var_79);
    // wp.dot(p1, axis0),                                                                     <L 399>
    var_82 = wp::dot(var_12, var_78);
    // wp.dot(p1, axis1),                                                                     <L 400>
    var_83 = wp::dot(var_12, var_79);
    var_84 = wp::vec_t<4, wp::float32>(var_80, var_81, var_82, var_83);
    // valid_side = wp.norm_l2(side) < MJ_MAXVAL                                              <L 404>
    var_85 = norm_l2_0(var_side);
    var_86 = (var_85 < var_5);
    // if valid_side:                                                                         <L 406>
    if (var_86) {
        // sidepnt = matT @ (side - pos)                                                      <L 408>
        var_87 = wp::sub(var_side, var_pos);
        var_88 = wp::mul(var_8, var_87);
        // sidepnt_proj = wp.vec2(                                                            <L 411>
        // wp.dot(sidepnt, axis0),                                                            <L 412>
        var_89 = wp::dot(var_88, var_78);
        // wp.dot(sidepnt, axis1),                                                            <L 413>
        var_90 = wp::dot(var_88, var_79);
        var_91 = wp::vec_t<2, wp::float32>(var_89, var_90);
        // sidepnt_proj, _ = math.normalize_with_norm(sidepnt_proj)                           <L 416>
        normalize_with_norm_0(var_91, var_92, var_93);
        // sidepnt_proj *= radius                                                             <L 417>
        var_94 = wp::mul(var_92, var_radius);
    }
    if (!var_86) {
        // sidepnt_proj = wp.vec2(MJ_MAXVAL)                                                  <L 419>
        var_95 = wp::vec_t<2, wp::float32>(var_5);
    }
    var_96 = wp::where(var_86, var_93, var_69);
    var_97 = wp::where(var_86, var_94, var_95);
    // if valid_side and wp.norm_l2(sidepnt) < radius:                                        <L 422>
    var_98 = var_86;
    if (var_98) {
        var_99 = norm_l2_0(var_88);
        var_100 = (var_99 < var_radius);
        var_98 = var_98 && var_100;
    }
    if (var_98) {
        // wlen, pnt0, pnt1 = wrap_inside(end, radius)                                        <L 423>
        wrap_inside_0(var_84, var_radius, var_104, var_105, var_106, var_101, var_102, var_103);
    }
    if (!var_98) {
        // wlen, pnt0, pnt1 = wrap_circle(end, sidepnt_proj, radius)                          <L 425>
        wrap_circle_0(var_84, var_97, var_radius, var_107, var_108, var_109);
    }
    var_110 = wp::where(var_98, var_101, var_107);
    var_111 = wp::where(var_98, var_102, var_108);
    var_112 = wp::where(var_98, var_103, var_109);
    // if wlen < 0.0:                                                                         <L 428>
    var_114 = (var_110 < var_113);
    if (var_114) {
        // return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                                <L 429>
        var_116 = wp::vec_t<3, wp::float32>(var_5);
        var_117 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_115;
        ret_1 = var_116;
        ret_2 = var_117;
        return;
    }
    // res0 = axis0 * pnt0[0] + axis1 * pnt0[1]                                               <L 432>
    var_119 = wp::extract(var_111, var_118);
    var_120 = wp::mul(var_78, var_119);
    var_122 = wp::extract(var_111, var_121);
    var_123 = wp::mul(var_79, var_122);
    var_124 = wp::add(var_120, var_123);
    // res1 = axis0 * pnt1[0] + axis1 * pnt1[1]                                               <L 433>
    var_126 = wp::extract(var_112, var_125);
    var_127 = wp::mul(var_78, var_126);
    var_129 = wp::extract(var_112, var_128);
    var_130 = wp::mul(var_79, var_129);
    var_131 = wp::add(var_127, var_130);
    // if geomtype == WrapType.CYLINDER:                                                      <L 436>
    var_133 = (var_geomtype == var_132);
    if (var_133) {
        // L0 = wp.sqrt((p0[0] - res0[0]) * (p0[0] - res0[0]) + (p0[1] - res0[1]) * (p0[1] - res0[1]))       <L 438>
        var_135 = wp::extract(var_10, var_134);
        var_137 = wp::extract(var_124, var_136);
        var_138 = wp::sub(var_135, var_137);
        var_140 = wp::extract(var_10, var_139);
        var_142 = wp::extract(var_124, var_141);
        var_143 = wp::sub(var_140, var_142);
        var_144 = wp::mul(var_138, var_143);
        var_146 = wp::extract(var_10, var_145);
        var_148 = wp::extract(var_124, var_147);
        var_149 = wp::sub(var_146, var_148);
        var_151 = wp::extract(var_10, var_150);
        var_153 = wp::extract(var_124, var_152);
        var_154 = wp::sub(var_151, var_153);
        var_155 = wp::mul(var_149, var_154);
        var_156 = wp::add(var_144, var_155);
        var_157 = wp::sqrt(var_156);
        // L1 = wp.sqrt((p1[0] - res1[0]) * (p1[0] - res1[0]) + (p1[1] - res1[1]) * (p1[1] - res1[1]))       <L 439>
        var_159 = wp::extract(var_12, var_158);
        var_161 = wp::extract(var_131, var_160);
        var_162 = wp::sub(var_159, var_161);
        var_164 = wp::extract(var_12, var_163);
        var_166 = wp::extract(var_131, var_165);
        var_167 = wp::sub(var_164, var_166);
        var_168 = wp::mul(var_162, var_167);
        var_170 = wp::extract(var_12, var_169);
        var_172 = wp::extract(var_131, var_171);
        var_173 = wp::sub(var_170, var_172);
        var_175 = wp::extract(var_12, var_174);
        var_177 = wp::extract(var_131, var_176);
        var_178 = wp::sub(var_175, var_177);
        var_179 = wp::mul(var_173, var_178);
        var_180 = wp::add(var_168, var_179);
        var_181 = wp::sqrt(var_180);
        // res0[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0, L0 + wlen + L1)              <L 440>
        var_183 = wp::extract(var_10, var_182);
        var_185 = wp::extract(var_12, var_184);
        var_187 = wp::extract(var_10, var_186);
        var_188 = wp::sub(var_185, var_187);
        var_189 = wp::add(var_157, var_110);
        var_190 = wp::add(var_189, var_181);
        var_191 = safe_div_0(var_157, var_190);
        var_192 = wp::mul(var_188, var_191);
        var_193 = wp::add(var_183, var_192);
        wp::assign_inplace(var_124, var_194, var_193);
        // res1[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0 + wlen, L0 + wlen + L1)       <L 441>
        var_196 = wp::extract(var_10, var_195);
        var_198 = wp::extract(var_12, var_197);
        var_200 = wp::extract(var_10, var_199);
        var_201 = wp::sub(var_198, var_200);
        var_202 = wp::add(var_157, var_110);
        var_203 = wp::add(var_157, var_110);
        var_204 = wp::add(var_203, var_181);
        var_205 = safe_div_0(var_202, var_204);
        var_206 = wp::mul(var_201, var_205);
        var_207 = wp::add(var_196, var_206);
        wp::assign_inplace(var_131, var_208, var_207);
        // height = wp.abs(res1[2] - res0[2])                                                 <L 444>
        var_210 = wp::extract(var_131, var_209);
        var_212 = wp::extract(var_124, var_211);
        var_213 = wp::sub(var_210, var_212);
        var_214 = wp::abs(var_213);
        // wlen = wp.sqrt(wlen * wlen + height * height)                                      <L 445>
        var_215 = wp::mul(var_110, var_110);
        var_216 = wp::mul(var_214, var_214);
        var_217 = wp::add(var_215, var_216);
        var_218 = wp::sqrt(var_217);
    }
    var_219 = wp::where(var_133, var_218, var_110);
    // wpnt0 = mat @ res0 + pos                                                               <L 448>
    var_220 = wp::mul(var_mat, var_124);
    var_221 = wp::add(var_220, var_pos);
    // wpnt1 = mat @ res1 + pos                                                               <L 449>
    var_222 = wp::mul(var_mat, var_131);
    var_223 = wp::add(var_222, var_pos);
    // return wlen, wpnt0, wpnt1                                                              <L 451>
    ret_0 = var_219;
    ret_1 = var_221;
    ret_2 = var_223;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/smooth.py:3693
static CUDA_CALLABLE void _accumulate_jac_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_J_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    bool var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    bool var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    wp::int32* var_28;
    bool var_29;
    wp::int32 var_30;
    wp::vec_t<6, wp::float32>* var_31;
    wp::vec_t<6, wp::float32> var_32;
    wp::vec_t<6, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::array_t<wp::float32> var_42;
    wp::float32 var_43;
    wp::int32* var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    //---------
    // forward
    // def _accumulate_jac_chain(                                                             <L 3694>
    // ptr = rownnz - 1                                                                       <L 3714>
    var_1 = wp::sub(var_rownnz, var_0);
    // bid = bodyid                                                                           <L 3715>
    var_2 = wp::copy(var_bodyid);
    // while bid > 0:                                                                         <L 3716>
    start_while_0:;
    var_4 = (var_2 > var_3);
    if ((var_4) == false) goto end_while_0;
        // bdofadr = body_dofadr[bid]                                                         <L 3717>
        var_5 = wp::address(var_body_dofadr, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // bdofnum = body_dofnum[bid]                                                         <L 3718>
        var_8 = wp::address(var_body_dofnum, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for k_rev in range(bdofnum):                                                       <L 3720>
        var_11 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_11) == 0) goto end_for_2;
            var_12 = wp::iter_next(var_11);
            // dof = bdofadr + bdofnum - 1 - k_rev                                            <L 3721>
            var_13 = wp::add(var_6, var_9);
            var_15 = wp::sub(var_13, var_14);
            var_16 = wp::sub(var_15, var_12);
            // while ptr >= 0:                                                                <L 3723>
    start_while_4:;
            var_18 = (var_1 >= var_17);
    if ((var_18) == false) goto end_while_4;
                // sparseid = rowadr + ptr                                                    <L 3724>
                var_19 = wp::add(var_rowadr, var_1);
                // if ten_J_colind[sparseid] <= dof:                                          <L 3725>
                var_20 = wp::address(var_ten_J_colind, var_19);
                var_22 = wp::load(var_20);
                var_21 = (var_22 <= var_16);
                if (var_21) {
                    // break                                                                  <L 3726>
                    goto end_while_4;
                }
                // ptr -= 1                                                                   <L 3727>
                var_24 = wp::sub(var_1, var_23);
                wp::assign(var_1, var_24);
    goto start_while_4;
    end_while_4:;
            // if ptr >= 0 and ten_J_colind[sparseid] == dof:                                 <L 3728>
            var_27 = (var_1 >= var_26);
            var_25 = var_27;
            if (var_25) {
                var_28 = wp::address(var_ten_J_colind, var_19);
                var_30 = wp::load(var_28);
                var_29 = (var_30 == var_16);
                var_25 = var_25 && var_29;
            }
            if (var_25) {
                // cdof = cdof_in[worldid, dof]                                               <L 3729>
                var_31 = wp::address(var_cdof_in, var_worldid, var_16);
                var_33 = wp::load(var_31);
                var_32 = wp::copy(var_33);
                // cdof_ang = wp.spatial_top(cdof)                                            <L 3730>
                var_34 = wp::spatial_top(var_32);
                // cdof_lin = wp.spatial_bottom(cdof)                                         <L 3731>
                var_35 = wp::spatial_bottom(var_32);
                // jacp = cdof_lin + wp.cross(cdof_ang, offset)                               <L 3732>
                var_36 = wp::cross(var_34, var_offset);
                var_37 = wp::add(var_35, var_36);
                // J = wp.dot(jacp, vec) * scale                                              <L 3733>
                var_38 = wp::dot(var_37, var_vec);
                var_39 = wp::mul(var_38, var_scale);
                // if J != 0.0:                                                               <L 3734>
                var_41 = (var_39 != var_40);
                if (var_41) {
                    // wp.atomic_add(ten_J_out[worldid], sparseid, J)                         <L 3735>
                    var_42 = wp::view(var_ten_J_out, var_worldid);
                    var_43 = wp::atomic_add(var_42, var_19, var_39);
                }
            }
            goto start_for_2;
        end_for_2:;
        // bid = body_parentid[bid]                                                           <L 3736>
        var_44 = wp::address(var_body_parentid, var_2);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        wp::assign(var_2, var_45);
    goto start_while_0;
    end_while_0:;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:147
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:       <L 148>
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 150>
    var_1 = wp::extract(var_v, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_5 = wp::extract(var_v, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 151>
    var_8 = wp::extract(var_v, var_7);
    var_10 = wp::extract(var_v, var_9);
    var_12 = wp::extract(var_v, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // f0 = wp.vec3(f[0], f[1], f[2])                                                         <L 152>
    var_15 = wp::extract(var_f, var_14);
    var_17 = wp::extract(var_f, var_16);
    var_19 = wp::extract(var_f, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // f1 = wp.vec3(f[3], f[4], f[5])                                                         <L 153>
    var_22 = wp::extract(var_f, var_21);
    var_24 = wp::extract(var_f, var_23);
    var_26 = wp::extract(var_f, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(v0, f0) + wp.cross(v1, f1)                                              <L 155>
    var_28 = wp::cross(var_6, var_20);
    var_29 = wp::cross(var_13, var_27);
    var_30 = wp::add(var_28, var_29);
    // vel = wp.cross(v0, f1)                                                                 <L 156>
    var_31 = wp::cross(var_6, var_27);
    // return wp.spatial_vector(ang, vel)                                                     <L 158>
    var_32 = wp::vec_t<6, wp::float32>(var_30, var_31);
    return var_32;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_inv_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    //---------
    // forward
    // def quat_inv(quat: wp.quat) -> wp.quat:                                                <L 116>
    // return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])                                  <L 117>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_quat, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_quat, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    return var_11;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::int32 var_4 = 3;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 2.0;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 3.141592653589793;
    bool var_18;
    const wp::float32 var_19 = 2.0;
    const wp::float32 var_20 = 3.141592653589793;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def quat_to_vel(quat: wp.quat) -> wp.vec3:                                             <L 162>
    // axis = wp.vec3(quat[1], quat[2], quat[3])                                              <L 163>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // sin_a_2 = wp.norm_l2(axis)                                                             <L 164>
    var_7 = norm_l2_0(var_6);
    // if sin_a_2 == 0.0:                                                                     <L 166>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return wp.vec3(0.0)                                                                <L 167>
        var_11 = wp::vec_t<3, wp::float32>(var_10);
        return var_11;
    }
    // speed = 2.0 * wp.atan2(sin_a_2, quat[0])                                               <L 169>
    var_14 = wp::extract(var_quat, var_13);
    var_15 = wp::atan2(var_7, var_14);
    var_16 = wp::mul(var_12, var_15);
    // if speed > wp.pi:                                                                      <L 171>
    var_18 = (var_16 > var_17);
    if (var_18) {
        // speed -= 2.0 * wp.pi                                                               <L 172>
        var_21 = wp::mul(var_19, var_20);
        var_22 = wp::sub(var_16, var_21);
    }
    var_23 = wp::where(var_18, var_22, var_16);
    // return axis * speed / sin_a_2                                                          <L 174>
    var_24 = wp::mul(var_6, var_23);
    var_25 = wp::div(var_24, var_7);
    return var_25;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE wp::vec_t<3, wp::float32> safe_div_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:497
static CUDA_CALLABLE void jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::int32* var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::int32 var_3;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<3, wp::float32> var_5;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::int32* var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::int32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<6, wp::float32>* var_14;
    wp::vec_t<6, wp::float32> var_15;
    wp::vec_t<6, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    //---------
    // forward
    // def jac_dof(                                                                           <L 498>
    // if body_isdofancestor[bodyid, dofid] == 0:                                             <L 513>
    var_0 = wp::address(var_body_isdofancestor, var_bodyid, var_dofid);
    var_3 = wp::load(var_0);
    var_2 = (var_3 == var_1);
    if (var_2) {
        // return wp.vec3(0.0), wp.vec3(0.0)                                                  <L 514>
        var_5 = wp::vec_t<3, wp::float32>(var_4);
        var_7 = wp::vec_t<3, wp::float32>(var_6);
        ret_0 = var_5;
        ret_1 = var_7;
        return;
    }
    // offset = point - wp.vec3(subtree_com_in[worldid, body_rootid[bodyid]])                 <L 516>
    var_8 = wp::address(var_body_rootid, var_bodyid);
    var_10 = wp::load(var_8);
    var_9 = wp::address(var_subtree_com_in, var_worldid, var_10);
    var_12 = wp::load(var_9);
    var_11 = wp::vec_t<3, wp::float32>(var_12);
    var_13 = wp::sub(var_point, var_11);
    // cdof = cdof_in[worldid, dofid]                                                         <L 518>
    var_14 = wp::address(var_cdof_in, var_worldid, var_dofid);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // cdof_ang = wp.spatial_top(cdof)                                                        <L 519>
    var_17 = wp::spatial_top(var_15);
    // cdof_lin = wp.spatial_bottom(cdof)                                                     <L 520>
    var_18 = wp::spatial_bottom(var_15);
    // jacp = cdof_lin + wp.cross(cdof_ang, offset)                                           <L 522>
    var_19 = wp::cross(var_17, var_13);
    var_20 = wp::add(var_18, var_19);
    // jacr = cdof_ang                                                                        <L 523>
    var_21 = wp::copy(var_17);
    // return jacp, jacr                                                                      <L 525>
    ret_0 = var_20;
    ret_1 = var_21;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    wp::quat_t<wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:                                     <L 178>
    // qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])                                          <L 181>
    var_1 = wp::extract(var_qb, var_0);
    var_3 = wp::extract(var_qb, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_qb, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_qb, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    // qdif = mul_quat(qneg, qa)                                                              <L 182>
    var_12 = mul_quat_0(var_11, var_qa);
    // return quat_to_vel(qdif)                                                               <L 185>
    var_13 = quat_to_vel_0(var_12);
    return var_13;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/smooth.py:1825
static CUDA_CALLABLE void _accumulate_jac_dot_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_pvel_lin,
    wp::vec_t<3, wp::float32> var_dpnt,
    wp::vec_t<3, wp::float32> var_dvel,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_Jdot_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    bool var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    bool var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    wp::int32* var_28;
    bool var_29;
    wp::int32 var_30;
    wp::vec_t<6, wp::float32>* var_31;
    wp::vec_t<6, wp::float32> var_32;
    wp::vec_t<6, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<6, wp::float32>* var_36;
    wp::vec_t<6, wp::float32> var_37;
    wp::vec_t<6, wp::float32> var_38;
    wp::int32* var_39;
    wp::int32 var_40;
    wp::int32 var_41;
    wp::int32* var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32* var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    bool var_48;
    const wp::int32 var_49 = 1;
    bool var_50;
    bool var_51;
    const wp::int32 var_52 = 0;
    bool var_53;
    const wp::int32 var_54 = 3;
    wp::int32 var_55;
    bool var_56;
    wp::vec_t<6, wp::float32>* var_57;
    wp::vec_t<6, wp::float32> var_58;
    wp::vec_t<6, wp::float32> var_59;
    wp::vec_t<6, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::float32 var_73 = 0.0;
    bool var_74;
    wp::array_t<wp::float32> var_75;
    wp::float32 var_76;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    //---------
    // forward
    // def _accumulate_jac_dot_chain(                                                         <L 1826>
    // ptr = rownnz - 1                                                                       <L 1853>
    var_1 = wp::sub(var_rownnz, var_0);
    // bid = bodyid                                                                           <L 1854>
    var_2 = wp::copy(var_bodyid);
    // while bid > 0:                                                                         <L 1855>
    start_while_0:;
    var_4 = (var_2 > var_3);
    if ((var_4) == false) goto end_while_0;
        // bdofadr = body_dofadr[bid]                                                         <L 1856>
        var_5 = wp::address(var_body_dofadr, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // bdofnum = body_dofnum[bid]                                                         <L 1857>
        var_8 = wp::address(var_body_dofnum, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for k_rev in range(bdofnum):                                                       <L 1859>
        var_11 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_11) == 0) goto end_for_2;
            var_12 = wp::iter_next(var_11);
            // dof = bdofadr + bdofnum - 1 - k_rev                                            <L 1860>
            var_13 = wp::add(var_6, var_9);
            var_15 = wp::sub(var_13, var_14);
            var_16 = wp::sub(var_15, var_12);
            // while ptr >= 0:                                                                <L 1862>
    start_while_4:;
            var_18 = (var_1 >= var_17);
    if ((var_18) == false) goto end_while_4;
                // sparseid = rowadr + ptr                                                    <L 1863>
                var_19 = wp::add(var_rowadr, var_1);
                // if ten_J_colind[sparseid] <= dof:                                          <L 1864>
                var_20 = wp::address(var_ten_J_colind, var_19);
                var_22 = wp::load(var_20);
                var_21 = (var_22 <= var_16);
                if (var_21) {
                    // break                                                                  <L 1865>
                    goto end_while_4;
                }
                // ptr -= 1                                                                   <L 1866>
                var_24 = wp::sub(var_1, var_23);
                wp::assign(var_1, var_24);
    goto start_while_4;
    end_while_4:;
            // if ptr >= 0 and ten_J_colind[sparseid] == dof:                                 <L 1867>
            var_27 = (var_1 >= var_26);
            var_25 = var_27;
            if (var_25) {
                var_28 = wp::address(var_ten_J_colind, var_19);
                var_30 = wp::load(var_28);
                var_29 = (var_30 == var_16);
                var_25 = var_25 && var_29;
            }
            if (var_25) {
                // cdof = cdof_in[worldid, dof]                                               <L 1868>
                var_31 = wp::address(var_cdof_in, var_worldid, var_16);
                var_33 = wp::load(var_31);
                var_32 = wp::copy(var_33);
                // cdof_ang = wp.spatial_top(cdof)                                            <L 1869>
                var_34 = wp::spatial_top(var_32);
                // cdof_lin = wp.spatial_bottom(cdof)                                         <L 1870>
                var_35 = wp::spatial_bottom(var_32);
                // cdof_dot = cdof_dot_in[worldid, dof]                                       <L 1871>
                var_36 = wp::address(var_cdof_dot_in, var_worldid, var_16);
                var_38 = wp::load(var_36);
                var_37 = wp::copy(var_38);
                // dofjntid = dof_jntid[dof]                                                  <L 1874>
                var_39 = wp::address(var_dof_jntid, var_16);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // jnttype = jnt_type[dofjntid]                                               <L 1875>
                var_42 = wp::address(var_jnt_type, var_40);
                var_44 = wp::load(var_42);
                var_43 = wp::copy(var_44);
                // jntdofadr = jnt_dofadr[dofjntid]                                           <L 1876>
                var_45 = wp::address(var_jnt_dofadr, var_40);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // if (jnttype == JointType.BALL) or ((jnttype == JointType.FREE) and dof >= jntdofadr + 3):       <L 1877>
                var_50 = (var_43 == var_49);
                var_48 = var_50;
                if (!var_48) {
                    var_53 = (var_43 == var_52);
                    var_51 = var_53;
                    if (var_51) {
                        var_55 = wp::add(var_46, var_54);
                        var_56 = (var_16 >= var_55);
                        var_51 = var_51 && var_56;
                    }
                    var_48 = var_48 || var_51;
                }
                if (var_48) {
                    // cdof_dot = math.motion_cross(cvel_in[worldid, bid], cdof)              <L 1878>
                    var_57 = wp::address(var_cvel_in, var_worldid, var_2);
                    var_59 = wp::load(var_57);
                    var_58 = motion_cross_0(var_59, var_32);
                }
                var_60 = wp::where(var_48, var_58, var_37);
                // cdof_dot_ang = wp.spatial_top(cdof_dot)                                    <L 1880>
                var_61 = wp::spatial_top(var_60);
                // cdof_dot_lin = wp.spatial_bottom(cdof_dot)                                 <L 1881>
                var_62 = wp::spatial_bottom(var_60);
                // jacp_dot = cdof_dot_lin + wp.cross(cdof_dot_ang, offset) + wp.cross(cdof_ang, pvel_lin)       <L 1884>
                var_63 = wp::cross(var_61, var_offset);
                var_64 = wp::add(var_62, var_63);
                var_65 = wp::cross(var_34, var_pvel_lin);
                var_66 = wp::add(var_64, var_65);
                // jacp = cdof_lin + wp.cross(cdof_ang, offset)                               <L 1887>
                var_67 = wp::cross(var_34, var_offset);
                var_68 = wp::add(var_35, var_67);
                // Jdot = (wp.dot(jacp_dot, dpnt) + wp.dot(jacp, dvel)) * scale               <L 1890>
                var_69 = wp::dot(var_66, var_dpnt);
                var_70 = wp::dot(var_68, var_dvel);
                var_71 = wp::add(var_69, var_70);
                var_72 = wp::mul(var_71, var_scale);
                // if Jdot != 0.0:                                                            <L 1891>
                var_74 = (var_72 != var_73);
                if (var_74) {
                    // wp.atomic_add(ten_Jdot_out[worldid], sparseid, Jdot)                   <L 1892>
                    var_75 = wp::view(var_ten_Jdot_out, var_worldid);
                    var_76 = wp::atomic_add(var_75, var_19, var_72);
                }
            }
            goto start_for_2;
        end_for_2:;
        // bid = body_parentid[bid]                                                           <L 1893>
        var_77 = wp::address(var_body_parentid, var_2);
        var_79 = wp::load(var_77);
        var_78 = wp::copy(var_79);
        wp::assign(var_2, var_78);
    goto start_while_0;
    end_while_0:;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:984
static CUDA_CALLABLE wp::float32 _phi_0(
    wp::float32 var_s,
    wp::int32 var_i)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 1.0;
    wp::float32 var_3;
    //---------
    // forward
    // def _phi(s: float, i: int) -> float:                                                   <L 985>
    // if i == 0:                                                                             <L 991>
    var_1 = (var_i == var_0);
    if (var_1) {
        // return 1.0 - s                                                                     <L 992>
        var_3 = wp::sub(var_2, var_s);
        return var_3;
    }
    // return s                                                                               <L 993>
    return var_s;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:996
static CUDA_CALLABLE wp::float32 eval_basis_trilinear_0(
    wp::vec_t<3, wp::float32> var_local,
    wp::int32 var_node_idx)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    //---------
    // forward
    // def eval_basis_trilinear(local: wp.vec3, node_idx: int) -> float:                      <L 997>
    // k = node_idx & 1                                                                       <L 1003>
    var_1 = wp::bit_and(var_node_idx, var_0);
    // j = (node_idx >> 1) & 1                                                                <L 1004>
    var_3 = wp::rshift(var_node_idx, var_2);
    var_5 = wp::bit_and(var_3, var_4);
    // i = (node_idx >> 2) & 1                                                                <L 1005>
    var_7 = wp::rshift(var_node_idx, var_6);
    var_9 = wp::bit_and(var_7, var_8);
    // return _phi(local[0], i) * _phi(local[1], j) * _phi(local[2], k)                       <L 1006>
    var_11 = wp::extract(var_local, var_10);
    var_12 = _phi_0(var_11, var_9);
    var_14 = wp::extract(var_local, var_13);
    var_15 = _phi_0(var_14, var_5);
    var_16 = wp::mul(var_12, var_15);
    var_18 = wp::extract(var_local, var_17);
    var_19 = _phi_0(var_18, var_1);
    var_20 = wp::mul(var_16, var_19);
    return var_20;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE wp::quat_t<wp::float32> axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::quat_t<wp::float32> var_13;
    //---------
    // forward
    // def axis_angle_to_quat(axis: wp.vec3, angle: float) -> wp.quat:                        <L 53>
    // s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)                                        <L 54>
    var_1 = wp::mul(var_angle, var_0);
    var_2 = wp::sin(var_1);
    var_4 = wp::mul(var_angle, var_3);
    var_5 = wp::cos(var_4);
    // axis = axis * s                                                                        <L 55>
    var_6 = wp::mul(var_axis, var_2);
    // return wp.quat(c, axis[0], axis[1], axis[2])                                           <L 56>
    var_8 = wp::extract(var_6, var_7);
    var_10 = wp::extract(var_6, var_9);
    var_12 = wp::extract(var_6, var_11);
    var_13 = wp::quat_t<wp::float32>(var_5, var_8, var_10, var_12);
    return var_13;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:1065
static CUDA_CALLABLE void get_face_metadata_0(
    wp::int32 var_cellnum_x,
    wp::int32 var_cellnum_y,
    wp::int32 var_cellnum_z,
    wp::int32 var_face_elem_idx,
    wp::int32 var_order_abs,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    wp::int32 & ret_2,
    wp::int32 & ret_3,
    wp::int32 & ret_4,
    wp::int32 & ret_5)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    bool var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    const wp::int32 var_13 = 2;
    wp::int32 var_14;
    wp::int32 var_15;
    bool var_16;
    const wp::int32 var_17 = 2;
    const wp::int32 var_18 = 2;
    wp::int32 var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = 2;
    wp::int32 var_22;
    const wp::int32 var_23 = 2;
    wp::int32 var_24;
    wp::int32 var_25;
    bool var_26;
    const wp::int32 var_27 = 3;
    const wp::int32 var_28 = 2;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    const wp::int32 var_32 = 2;
    wp::int32 var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::int32 var_37;
    bool var_38;
    const wp::int32 var_39 = 4;
    const wp::int32 var_40 = 2;
    wp::int32 var_41;
    wp::int32 var_42;
    const wp::int32 var_43 = 2;
    wp::int32 var_44;
    wp::int32 var_45;
    const wp::int32 var_46 = 5;
    const wp::int32 var_47 = 2;
    wp::int32 var_48;
    wp::int32 var_49;
    const wp::int32 var_50 = 2;
    wp::int32 var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    wp::int32 var_56;
    wp::int32 var_57;
    wp::int32 var_58;
    wp::int32 var_59;
    wp::int32 var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    const wp::int32 var_64 = 2;
    wp::int32 var_65;
    const wp::int32 var_66 = 0;
    bool var_67;
    const wp::int32 var_68 = 0;
    bool var_69;
    const wp::int32 var_70 = 1;
    bool var_71;
    wp::int32 var_72;
    bool var_73;
    const wp::int32 var_74 = 2;
    bool var_75;
    const wp::int32 var_76 = 3;
    bool var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::int32 var_80;
    wp::int32 var_81;
    const wp::int32 var_82 = 0;
    const wp::int32 var_83 = 0;
    bool var_84;
    wp::int32 var_85;
    const wp::int32 var_86 = 1;
    bool var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    wp::int32 var_90;
    wp::int32 var_91;
    const wp::int32 var_92 = 2;
    wp::int32 var_93;
    wp::int32 var_94;
    wp::int32 var_95;
    wp::int32 var_96;
    wp::int32 var_97;
    wp::int32 var_98;
    const wp::int32 var_99 = 1;
    wp::int32 var_100;
    wp::int32 var_101;
    const wp::int32 var_102 = 1;
    wp::int32 var_103;
    //---------
    // forward
    // def get_face_metadata(                                                                 <L 1066>
    // size01 = cellnum_y * cellnum_z                                                         <L 1074>
    var_0 = wp::mul(var_cellnum_y, var_cellnum_z);
    // size23 = cellnum_x * cellnum_z                                                         <L 1075>
    var_1 = wp::mul(var_cellnum_x, var_cellnum_z);
    // size45 = cellnum_x * cellnum_y                                                         <L 1076>
    var_2 = wp::mul(var_cellnum_x, var_cellnum_y);
    // face_id = 0                                                                            <L 1078>
    // within_face = 0                                                                        <L 1079>
    // if face_elem_idx < size01:                                                             <L 1081>
    var_5 = (var_face_elem_idx < var_0);
    if (var_5) {
        // face_id = 0                                                                        <L 1082>
        // within_face = face_elem_idx                                                        <L 1083>
        var_7 = wp::copy(var_face_elem_idx);
    }
    if (!var_5) {
        // elif face_elem_idx < 2 * size01:                                                   <L 1084>
        var_9 = wp::mul(var_8, var_0);
        var_10 = (var_face_elem_idx < var_9);
        if (var_10) {
            // face_id = 1                                                                    <L 1085>
            // within_face = face_elem_idx - size01                                           <L 1086>
            var_12 = wp::sub(var_face_elem_idx, var_0);
        }
        if (!var_10) {
            // elif face_elem_idx < 2 * size01 + size23:                                      <L 1087>
            var_14 = wp::mul(var_13, var_0);
            var_15 = wp::add(var_14, var_1);
            var_16 = (var_face_elem_idx < var_15);
            if (var_16) {
                // face_id = 2                                                                <L 1088>
                // within_face = face_elem_idx - 2 * size01                                   <L 1089>
                var_19 = wp::mul(var_18, var_0);
                var_20 = wp::sub(var_face_elem_idx, var_19);
            }
            if (!var_16) {
                // elif face_elem_idx < 2 * size01 + 2 * size23:                              <L 1090>
                var_22 = wp::mul(var_21, var_0);
                var_24 = wp::mul(var_23, var_1);
                var_25 = wp::add(var_22, var_24);
                var_26 = (var_face_elem_idx < var_25);
                if (var_26) {
                    // face_id = 3                                                            <L 1091>
                    // within_face = face_elem_idx - 2 * size01 - size23                      <L 1092>
                    var_29 = wp::mul(var_28, var_0);
                    var_30 = wp::sub(var_face_elem_idx, var_29);
                    var_31 = wp::sub(var_30, var_1);
                }
                if (!var_26) {
                    // elif face_elem_idx < 2 * size01 + 2 * size23 + size45:                 <L 1093>
                    var_33 = wp::mul(var_32, var_0);
                    var_35 = wp::mul(var_34, var_1);
                    var_36 = wp::add(var_33, var_35);
                    var_37 = wp::add(var_36, var_2);
                    var_38 = (var_face_elem_idx < var_37);
                    if (var_38) {
                        // face_id = 4                                                        <L 1094>
                        // within_face = face_elem_idx - 2 * size01 - 2 * size23              <L 1095>
                        var_41 = wp::mul(var_40, var_0);
                        var_42 = wp::sub(var_face_elem_idx, var_41);
                        var_44 = wp::mul(var_43, var_1);
                        var_45 = wp::sub(var_42, var_44);
                    }
                    if (!var_38) {
                        // face_id = 5                                                        <L 1097>
                        // within_face = face_elem_idx - 2 * size01 - 2 * size23 - size45       <L 1098>
                        var_48 = wp::mul(var_47, var_0);
                        var_49 = wp::sub(var_face_elem_idx, var_48);
                        var_51 = wp::mul(var_50, var_1);
                        var_52 = wp::sub(var_49, var_51);
                        var_53 = wp::sub(var_52, var_2);
                    }
                    var_54 = wp::where(var_38, var_39, var_46);
                    var_55 = wp::where(var_38, var_45, var_53);
                }
                var_56 = wp::where(var_26, var_27, var_54);
                var_57 = wp::where(var_26, var_31, var_55);
            }
            var_58 = wp::where(var_16, var_17, var_56);
            var_59 = wp::where(var_16, var_20, var_57);
        }
        var_60 = wp::where(var_10, var_11, var_58);
        var_61 = wp::where(var_10, var_12, var_59);
    }
    var_62 = wp::where(var_5, var_6, var_60);
    var_63 = wp::where(var_5, var_7, var_61);
    // normal_axis = face_id // 2                                                             <L 1100>
    var_65 = wp::floordiv(var_62, var_64);
    // c1 = 0                                                                                 <L 1102>
    // if face_id == 0 or face_id == 1:                                                       <L 1103>
    var_69 = (var_62 == var_68);
    var_67 = var_69;
    if (!var_67) {
        var_71 = (var_62 == var_70);
        var_67 = var_67 || var_71;
    }
    if (var_67) {
        // c1 = cellnum_z                                                                     <L 1104>
        var_72 = wp::copy(var_cellnum_z);
    }
    if (!var_67) {
        // elif face_id == 2 or face_id == 3:                                                 <L 1105>
        var_75 = (var_62 == var_74);
        var_73 = var_75;
        if (!var_73) {
            var_77 = (var_62 == var_76);
            var_73 = var_73 || var_77;
        }
        if (var_73) {
            // c1 = cellnum_x                                                                 <L 1106>
            var_78 = wp::copy(var_cellnum_x);
        }
        if (!var_73) {
            // c1 = cellnum_y                                                                 <L 1108>
            var_79 = wp::copy(var_cellnum_y);
        }
        var_80 = wp::where(var_73, var_78, var_79);
    }
    var_81 = wp::where(var_67, var_72, var_80);
    // fixed_dim = 0                                                                          <L 1110>
    // if normal_axis == 0:                                                                   <L 1111>
    var_84 = (var_65 == var_83);
    if (var_84) {
        // fixed_dim = cellnum_x                                                              <L 1112>
        var_85 = wp::copy(var_cellnum_x);
    }
    if (!var_84) {
        // elif normal_axis == 1:                                                             <L 1113>
        var_87 = (var_65 == var_86);
        if (var_87) {
            // fixed_dim = cellnum_y                                                          <L 1114>
            var_88 = wp::copy(var_cellnum_y);
        }
        if (!var_87) {
            // fixed_dim = cellnum_z                                                          <L 1116>
            var_89 = wp::copy(var_cellnum_z);
        }
        var_90 = wp::where(var_87, var_88, var_89);
    }
    var_91 = wp::where(var_84, var_85, var_90);
    // g_fixed = (face_id % 2) * fixed_dim * order_abs                                        <L 1117>
    var_93 = wp::mod(var_62, var_92);
    var_94 = wp::mul(var_93, var_91);
    var_95 = wp::mul(var_94, var_order_abs);
    // q0 = within_face // c1                                                                 <L 1119>
    var_96 = wp::floordiv(var_63, var_81);
    // q1 = within_face % c1                                                                  <L 1120>
    var_97 = wp::mod(var_63, var_81);
    // ny_g = cellnum_y * order_abs + 1                                                       <L 1122>
    var_98 = wp::mul(var_cellnum_y, var_order_abs);
    var_100 = wp::add(var_98, var_99);
    // nz_g = cellnum_z * order_abs + 1                                                       <L 1123>
    var_101 = wp::mul(var_cellnum_z, var_order_abs);
    var_103 = wp::add(var_101, var_102);
    // return normal_axis, g_fixed, q0, q1, ny_g, nz_g                                        <L 1125>
    ret_0 = var_65;
    ret_1 = var_95;
    ret_2 = var_96;
    ret_3 = var_97;
    ret_4 = var_100;
    ret_5 = var_103;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:67
static CUDA_CALLABLE wp::quat_t<wp::float32> mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::quat_t<wp::float32> var_4;
    const wp::int32 var_5 = 50;
    wp::range_t var_6;
    wp::int32 var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 1;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 2;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 1;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 2;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 1e-10;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 1e-06;
    bool var_39;
    wp::vec_t<3, wp::float32> var_40;
    const wp::float32 var_41 = 0.5;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::quat_t<wp::float32> var_56;
    wp::quat_t<wp::float32> var_57;
    wp::quat_t<wp::float32> var_58;
    //---------
    // forward
    // def mat33_to_quat_polar(F: wp.mat33) -> wp.quat:                                       <L 68>
    // cell_quat = wp.quat(0.0, 0.0, 0.0, 1.0)                                                <L 69>
    var_4 = wp::quat_t<wp::float32>(var_0, var_1, var_2, var_3);
    // for _iter in range(50):                                                                <L 70>
    var_6 = wp::range(var_5);
    start_for_0:;
        if (iter_cmp(var_6) == 0) goto end_for_0;
        var_7 = wp::iter_next(var_6);
        // rot = wp.quat_to_matrix(cell_quat)                                                 <L 71>
        var_8 = wp::quat_to_matrix(var_4);
        // rot_t = wp.transpose(rot)                                                          <L 72>
        var_9 = wp::transpose(var_8);
        // col1_rot = rot_t[0]                                                                <L 73>
        var_11 = wp::extract(var_9, var_10);
        // col2_rot = rot_t[1]                                                                <L 74>
        var_13 = wp::extract(var_9, var_12);
        // col3_rot = rot_t[2]                                                                <L 75>
        var_15 = wp::extract(var_9, var_14);
        // F_t = wp.transpose(F)                                                              <L 76>
        var_16 = wp::transpose(var_F);
        // col1_mat = F_t[0]                                                                  <L 77>
        var_18 = wp::extract(var_16, var_17);
        // col2_mat = F_t[1]                                                                  <L 78>
        var_20 = wp::extract(var_16, var_19);
        // col3_mat = F_t[2]                                                                  <L 79>
        var_22 = wp::extract(var_16, var_21);
        // omega = wp.cross(col1_rot, col1_mat) + wp.cross(col2_rot, col2_mat) + wp.cross(col3_rot, col3_mat)       <L 81>
        var_23 = wp::cross(var_11, var_18);
        var_24 = wp::cross(var_13, var_20);
        var_25 = wp::add(var_23, var_24);
        var_26 = wp::cross(var_15, var_22);
        var_27 = wp::add(var_25, var_26);
        // denom = wp.abs(wp.dot(col1_rot, col1_mat) + wp.dot(col2_rot, col2_mat) + wp.dot(col3_rot, col3_mat)) + 1.0e-10       <L 82>
        var_28 = wp::dot(var_11, var_18);
        var_29 = wp::dot(var_13, var_20);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::dot(var_15, var_22);
        var_32 = wp::add(var_30, var_31);
        var_33 = wp::abs(var_32);
        var_35 = wp::add(var_33, var_34);
        // omega = omega / denom                                                              <L 83>
        var_36 = wp::div(var_27, var_35);
        // w = wp.length(omega)                                                               <L 85>
        var_37 = wp::length(var_36);
        // if w < 1.0e-6:                                                                     <L 86>
        var_39 = (var_37 < var_38);
        if (var_39) {
            // break                                                                          <L 87>
            goto end_for_0;
        }
        // axis = omega / w                                                                   <L 89>
        var_40 = wp::div(var_36, var_37);
        // half_w = 0.5 * w                                                                   <L 90>
        var_42 = wp::mul(var_41, var_37);
        // qrot = wp.quat(                                                                    <L 91>
        // axis[0] * wp.sin(half_w),                                                          <L 92>
        var_44 = wp::extract(var_40, var_43);
        var_45 = wp::sin(var_42);
        var_46 = wp::mul(var_44, var_45);
        // axis[1] * wp.sin(half_w),                                                          <L 93>
        var_48 = wp::extract(var_40, var_47);
        var_49 = wp::sin(var_42);
        var_50 = wp::mul(var_48, var_49);
        // axis[2] * wp.sin(half_w),                                                          <L 94>
        var_52 = wp::extract(var_40, var_51);
        var_53 = wp::sin(var_42);
        var_54 = wp::mul(var_52, var_53);
        // wp.cos(half_w),                                                                    <L 95>
        var_55 = wp::cos(var_42);
        var_56 = wp::quat_t<wp::float32>(var_46, var_50, var_54, var_55);
        // cell_quat = wp.normalize(qrot * cell_quat)                                         <L 97>
        var_57 = wp::mul(var_56, var_4);
        var_58 = wp::normalize(var_57);
        wp::assign(var_4, var_58);
        goto start_for_0;
    end_for_0:;
    // return cell_quat                                                                       <L 98>
    return var_4;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:325
static CUDA_CALLABLE wp::vec_t<6, wp::float32> _decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim)
{
    //---------
    // primal vars
    wp::vec_t<6, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    bool var_2;
    wp::float32* var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    const wp::int32 var_13 = 2;
    wp::int32 var_14;
    wp::int32 var_15;
    bool var_16;
    wp::float32* var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 0.0;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    bool var_24;
    const wp::int32 var_25 = 1;
    wp::int32 var_26;
    wp::float32* var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    //---------
    // forward
    // def _decode_pyramid(njmax_in: int, pyramid: wp.array[float], efc_address: int, mu: vec5, condim: int) -> wp.spatial_vector:       <L 326>
    // force = wp.spatial_vector()                                                            <L 328>
    var_0 = wp::vec_t<6, wp::float32>();
    // if condim == 1:                                                                        <L 330>
    var_2 = (var_condim == var_1);
    if (var_2) {
        // force[0] = pyramid[efc_address]                                                    <L 331>
        var_3 = wp::address(var_pyramid, var_efc_address);
        var_5 = wp::load(var_3);
        wp::assign_inplace(var_0, var_4, var_5);
        // return force                                                                       <L 332>
        return var_0;
    }
    // force[0] = float(0.0)                                                                  <L 334>
    var_7 = wp::float(var_6);
    wp::assign_inplace(var_0, var_8, var_7);
    // for i in range(condim - 1):                                                            <L 335>
    var_10 = wp::sub(var_condim, var_9);
    var_11 = wp::range(var_10);
    start_for_1:;
        if (iter_cmp(var_11) == 0) goto end_for_1;
        var_12 = wp::iter_next(var_11);
        // adr = 2 * i + efc_address                                                          <L 336>
        var_14 = wp::mul(var_13, var_12);
        var_15 = wp::add(var_14, var_efc_address);
        // if adr < njmax_in:                                                                 <L 337>
        var_16 = (var_15 < var_njmax_in);
        if (var_16) {
            // dir1 = pyramid[adr]                                                            <L 338>
            var_17 = wp::address(var_pyramid, var_15);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        if (!var_16) {
            // dir1 = 0.0                                                                     <L 340>
        }
        var_21 = wp::where(var_16, var_18, var_20);
        // if adr + 1 < njmax_in:                                                             <L 341>
        var_23 = wp::add(var_15, var_22);
        var_24 = (var_23 < var_njmax_in);
        if (var_24) {
            // dir2 = pyramid[adr + 1]                                                        <L 342>
            var_26 = wp::add(var_15, var_25);
            var_27 = wp::address(var_pyramid, var_26);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
        }
        if (!var_24) {
            // dir2 = 0.0                                                                     <L 344>
        }
        var_31 = wp::where(var_24, var_28, var_30);
        // force[0] += dir1 + dir2                                                            <L 345>
        var_32 = wp::add(var_21, var_31);
        wp::add_inplace(var_0, var_33, var_32);
        // force[i + 1] = (dir1 - dir2) * mu[i]                                               <L 346>
        var_34 = wp::sub(var_21, var_31);
        var_35 = wp::extract(var_mu, var_12);
        var_36 = wp::mul(var_34, var_35);
        var_38 = wp::add(var_12, var_37);
        wp::assign_inplace(var_0, var_38, var_36);
        goto start_for_1;
    end_for_1:;
    // return force                                                                           <L 348>
    return var_0;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:351
static CUDA_CALLABLE wp::vec_t<6, wp::float32> contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<6, wp::float32> var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    bool var_14;
    const wp::int32 var_15 = 0;
    bool var_16;
    const wp::int32 var_17 = 0;
    wp::int32* var_18;
    bool var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = 0;
    bool var_22;
    const wp::int32 var_23 = 0;
    bool var_24;
    wp::array_t<wp::float32> var_25;
    wp::vec_t<5, wp::float32>* var_26;
    wp::vec_t<6, wp::float32> var_27;
    wp::vec_t<5, wp::float32> var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::int32* var_31;
    bool var_32;
    wp::int32 var_33;
    wp::int32* var_34;
    wp::float32* var_35;
    wp::int32 var_36;
    wp::float32 var_37;
    wp::vec_t<6, wp::float32> var_38;
    wp::vec_t<6, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::mat_t<3, 3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32>* var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::mat_t<3, 3, wp::float32> var_47;
    wp::vec_t<6, wp::float32> var_48;
    wp::vec_t<6, wp::float32> var_49;
    //---------
    // forward
    // def contact_force_fn(                                                                  <L 352>
    // force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                                <L 369>
    var_6 = wp::vec_t<6, wp::float32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // condim = contact_dim_in[contact_id]                                                    <L 370>
    var_7 = wp::address(var_contact_dim_in, var_contact_id);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // efc_address = contact_efc_address_in[contact_id, 0]                                    <L 371>
    var_11 = wp::address(var_contact_efc_address_in, var_contact_id, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // if contact_id >= 0 and contact_id <= nacon_in[0] and efc_address >= 0:                 <L 373>
    var_16 = (var_contact_id >= var_15);
    var_14 = var_16;
    if (var_14) {
        var_18 = wp::address(var_nacon_in, var_17);
        var_20 = wp::load(var_18);
        var_19 = (var_contact_id <= var_20);
        var_14 = var_14 && var_19;
    }
    if (var_14) {
        var_22 = (var_12 >= var_21);
        var_14 = var_14 && var_22;
    }
    if (var_14) {
        // if opt_cone == ConeType.PYRAMIDAL:                                                 <L 374>
        var_24 = (var_opt_cone == var_23);
        if (var_24) {
            // force = _decode_pyramid(                                                       <L 375>
            // njmax_in,                                                                      <L 376>
            // efc_force_in[worldid],                                                         <L 377>
            var_25 = wp::view(var_efc_force_in, var_worldid);
            // efc_address,                                                                   <L 378>
            // contact_friction_in[contact_id],                                               <L 379>
            var_26 = wp::address(var_contact_friction_in, var_contact_id);
            // condim,                                                                        <L 380>
            var_28 = wp::load(var_26);
            var_27 = _decode_pyramid_0(var_njmax_in, var_25, var_12, var_28, var_8);
        }
        if (!var_24) {
            // for i in range(condim):                                                        <L 383>
            var_29 = wp::range(var_8);
            start_for_0:;
                if (iter_cmp(var_29) == 0) goto end_for_0;
                var_30 = wp::iter_next(var_29);
                // if contact_efc_address_in[contact_id, i] < njmax_in:                       <L 384>
                var_31 = wp::address(var_contact_efc_address_in, var_contact_id, var_30);
                var_33 = wp::load(var_31);
                var_32 = (var_33 < var_njmax_in);
                if (var_32) {
                    // force[i] = efc_force_in[worldid, contact_efc_address_in[contact_id, i]]       <L 385>
                    var_34 = wp::address(var_contact_efc_address_in, var_contact_id, var_30);
                    var_36 = wp::load(var_34);
                    var_35 = wp::address(var_efc_force_in, var_worldid, var_36);
                    var_37 = wp::load(var_35);
                    wp::assign_inplace(var_6, var_30, var_37);
                }
                goto start_for_0;
            end_for_0:;
        }
        var_38 = wp::where(var_24, var_27, var_6);
    }
    var_39 = wp::where(var_14, var_38, var_6);
    // if to_world_frame:                                                                     <L 387>
    if (var_to_world_frame) {
        // t = wp.spatial_top(force) @ contact_frame_in[contact_id]                           <L 389>
        var_40 = wp::spatial_top(var_39);
        var_41 = wp::address(var_contact_frame_in, var_contact_id);
        var_43 = wp::load(var_41);
        var_42 = wp::mul(var_40, var_43);
        // b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]                        <L 390>
        var_44 = wp::spatial_bottom(var_39);
        var_45 = wp::address(var_contact_frame_in, var_contact_id);
        var_47 = wp::load(var_45);
        var_46 = wp::mul(var_44, var_47);
        // force = wp.spatial_vector(t, b)                                                    <L 391>
        var_48 = wp::vec_t<6, wp::float32>(var_42, var_46);
    }
    var_49 = wp::where(var_to_world_frame, var_48, var_39);
    // return force                                                                           <L 393>
    return var_49;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:469
static CUDA_CALLABLE wp::vec_t<6, wp::float32> transform_force_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::vec_t<3, wp::float32> var_torque,
    wp::vec_t<3, wp::float32> var_offset)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def transform_force(force: wp.vec3, torque: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:       <L 470>
    // return wp.spatial_vector(torque - wp.cross(offset, force), force)                      <L 471>
    var_0 = wp::cross(var_offset, var_force);
    var_1 = wp::sub(var_torque, var_0);
    var_2 = wp::vec_t<6, wp::float32>(var_1, var_force);
    return var_2;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:474
static CUDA_CALLABLE wp::vec_t<6, wp::float32> transform_force_1(
    wp::vec_t<6, wp::float32> var_frc,
    wp::vec_t<3, wp::float32> var_offset)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def transform_force(frc: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector:       <L 475>
    // force = wp.spatial_top(frc)                                                            <L 476>
    var_0 = wp::spatial_top(var_frc);
    // torque = wp.spatial_bottom(frc)                                                        <L 477>
    var_1 = wp::spatial_bottom(var_frc);
    // return transform_force(force, torque, offset)                                          <L 478>
    var_2 = transform_force_0(var_0, var_1, var_offset);
    return var_2;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:44
static CUDA_CALLABLE void adj_rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE void adj_mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v,
    wp::quat_t<wp::float32> & adj_u,
    wp::quat_t<wp::float32> & adj_v,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:59
static CUDA_CALLABLE void adj_quat_to_mat_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:133
static CUDA_CALLABLE void adj_motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> & adj_u,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:120
static CUDA_CALLABLE void adj_inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<10, wp::float32> & adj_i,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<2, wp::float32> var_x,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<2, wp::float32> & adj_x,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<2, wp::float32> var_v,
    wp::vec_t<2, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:202
static CUDA_CALLABLE void adj_wrap_inside_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::float32 var_radius,
    wp::int32 var_maxiter,
    wp::float32 var_zinit,
    wp::float32 var_tolerance,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<4, wp::float32> & adj_end,
    wp::float32 & adj_radius,
    wp::int32 & adj_maxiter,
    wp::float32 & adj_zinit,
    wp::float32 & adj_tolerance,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:31
static CUDA_CALLABLE void adj_is_intersect_0(
    wp::vec_t<2, wp::float32> var_p1,
    wp::vec_t<2, wp::float32> var_p2,
    wp::vec_t<2, wp::float32> var_p3,
    wp::vec_t<2, wp::float32> var_p4,
    wp::vec_t<2, wp::float32> & adj_p1,
    wp::vec_t<2, wp::float32> & adj_p2,
    wp::vec_t<2, wp::float32> & adj_p3,
    wp::vec_t<2, wp::float32> & adj_p4,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:77
static CUDA_CALLABLE void adj_length_circle_0(
    wp::vec_t<2, wp::float32> var_p0,
    wp::vec_t<2, wp::float32> var_p1,
    wp::int32 var_ind,
    wp::float32 var_radius,
    wp::vec_t<2, wp::float32> & adj_p0,
    wp::vec_t<2, wp::float32> & adj_p1,
    wp::int32 & adj_ind,
    wp::float32 & adj_radius,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:104
static CUDA_CALLABLE void adj_wrap_circle_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::vec_t<2, wp::float32> var_side,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<4, wp::float32> & adj_end,
    wp::vec_t<2, wp::float32> & adj_side,
    wp::float32 & adj_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/util_misc.py:327
static CUDA_CALLABLE void adj_wrap_0(
    wp::vec_t<3, wp::float32> var_x0,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::float32 var_radius,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_side,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_x0,
    wp::vec_t<3, wp::float32> & adj_x1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::float32 & adj_radius,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_side,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/smooth.py:3693
static CUDA_CALLABLE void adj__accumulate_jac_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_dofnum,
    wp::array_t<wp::int32> & adj_body_dofadr,
    wp::array_t<wp::int32> & adj_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rowadr,
    wp::int32 & adj_rownnz,
    wp::float32 & adj_scale,
    wp::int32 & adj_worldid,
    wp::array_t<wp::float32> & adj_ten_J_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:147
static CUDA_CALLABLE void adj_motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_f,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE void adj_quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::float32 var_y,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::float32 & adj_y,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:497
static CUDA_CALLABLE void adj_jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_dof_bodyid,
    wp::array_t<wp::int32> & adj_body_isdofancestor,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_point,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_dofid,
    wp::int32 & adj_worldid,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE void adj_quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb,
    wp::quat_t<wp::float32> & adj_qa,
    wp::quat_t<wp::float32> & adj_qb,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/smooth.py:1825
static CUDA_CALLABLE void adj__accumulate_jac_dot_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_pvel_lin,
    wp::vec_t<3, wp::float32> var_dpnt,
    wp::vec_t<3, wp::float32> var_dvel,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_Jdot_out,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_dofnum,
    wp::array_t<wp::int32> & adj_body_dofadr,
    wp::array_t<wp::int32> & adj_jnt_type,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::int32> & adj_dof_jntid,
    wp::array_t<wp::int32> & adj_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_dot_in,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<3, wp::float32> & adj_pvel_lin,
    wp::vec_t<3, wp::float32> & adj_dpnt,
    wp::vec_t<3, wp::float32> & adj_dvel,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rowadr,
    wp::int32 & adj_rownnz,
    wp::float32 & adj_scale,
    wp::int32 & adj_worldid,
    wp::array_t<wp::float32> & adj_ten_Jdot_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:984
static CUDA_CALLABLE void adj__phi_0(
    wp::float32 var_s,
    wp::int32 var_i,
    wp::float32 & adj_s,
    wp::int32 & adj_i,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:996
static CUDA_CALLABLE void adj_eval_basis_trilinear_0(
    wp::vec_t<3, wp::float32> var_local,
    wp::int32 var_node_idx,
    wp::vec_t<3, wp::float32> & adj_local,
    wp::int32 & adj_node_idx,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE void adj_axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::float32 & adj_angle,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:1065
static CUDA_CALLABLE void adj_get_face_metadata_0(
    wp::int32 var_cellnum_x,
    wp::int32 var_cellnum_y,
    wp::int32 var_cellnum_z,
    wp::int32 var_face_elem_idx,
    wp::int32 var_order_abs,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    wp::int32 & ret_2,
    wp::int32 & ret_3,
    wp::int32 & ret_4,
    wp::int32 & ret_5,
    wp::int32 & adj_cellnum_x,
    wp::int32 & adj_cellnum_y,
    wp::int32 & adj_cellnum_z,
    wp::int32 & adj_face_elem_idx,
    wp::int32 & adj_order_abs,
    wp::int32 & adj_ret_0,
    wp::int32 & adj_ret_1,
    wp::int32 & adj_ret_2,
    wp::int32 & adj_ret_3,
    wp::int32 & adj_ret_4,
    wp::int32 & adj_ret_5)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:67
static CUDA_CALLABLE void adj_mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F,
    wp::mat_t<3, 3, wp::float32> & adj_F,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:325
static CUDA_CALLABLE void adj__decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::float32> & adj_pyramid,
    wp::int32 & adj_efc_address,
    wp::vec_t<5, wp::float32> & adj_mu,
    wp::int32 & adj_condim,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:351
static CUDA_CALLABLE void adj_contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame,
    wp::int32 & adj_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_in,
    wp::array_t<wp::int32> & adj_contact_dim_in,
    wp::array_t<wp::int32> & adj_contact_efc_address_in,
    wp::array_t<wp::float32> & adj_efc_force_in,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::int32> & adj_nacon_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_contact_id,
    bool & adj_to_world_frame,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:469
static CUDA_CALLABLE void adj_transform_force_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::vec_t<3, wp::float32> var_torque,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> & adj_force,
    wp::vec_t<3, wp::float32> & adj_torque,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/support.py:474
static CUDA_CALLABLE void adj_transform_force_1(
    wp::vec_t<6, wp::float32> var_frc,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<6, wp::float32> & adj_frc,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _subtree_com_init_002b02fd_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _subtree_com_init(                                                                 <L 687>
        // worldid, bodyid = wp.tid()                                                             <L 695>
        builtin_tid2d(var_0, var_1);
        // subtree_com_out[worldid, bodyid] = xipos_in[worldid, bodyid] * body_mass[worldid % body_mass.shape[0], bodyid]       <L 696>
        var_2 = wp::address(var_xipos_in, var_0, var_1);
        var_3 = &(var_body_mass.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_body_mass, var_7, var_1);
        var_10 = wp::load(var_2);
        var_11 = wp::load(var_8);
        var_9 = wp::mul(var_10, var_11);
        wp::array_store(var_subtree_com_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _crb_accumulate_073a11b8_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::vec_t<10, wp::float32>* var_10;
        wp::vec_t<10, wp::float32> var_11;
        wp::vec_t<10, wp::float32> var_12;
        //---------
        // forward
        // def _crb_accumulate(                                                                   <L 1030>
        // worldid, nodeid = wp.tid()                                                             <L 1040>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 1041>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 1042>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if pid == 0:                                                                           <L 1043>
        var_9 = (var_6 == var_8);
        if (var_9) {
            // return                                                                             <L 1044>
            continue;
        }
        // wp.atomic_add(crb_out, worldid, pid, crb_in[worldid, bodyid])                          <L 1045>
        var_10 = wp::address(var_crb_in, var_0, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::atomic_add(var_crb_out, var_0, var_6, var_12);
    }
}



extern "C" __global__ void _site_local_to_global_084f92a4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::quat_t<wp::float32>* var_8;
        wp::quat_t<wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::shape_t* var_20;
        const wp::int32 var_21 = 0;
        wp::int32 var_22;
        wp::shape_t var_23;
        wp::int32 var_24;
        wp::quat_t<wp::float32>* var_25;
        wp::quat_t<wp::float32> var_26;
        wp::quat_t<wp::float32> var_27;
        wp::mat_t<3, 3, wp::float32> var_28;
        //---------
        // forward
        // def _site_local_to_global(                                                             <L 209>
        // worldid, siteid = wp.tid()                                                             <L 221>
        builtin_tid2d(var_0, var_1);
        // bodyid = site_bodyid[siteid]                                                           <L 222>
        var_2 = wp::address(var_site_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 223>
        var_5 = wp::address(var_xpos_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 224>
        var_8 = wp::address(var_xquat_in, var_0, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // site_xpos_out[worldid, siteid] = xpos + math.rot_vec_quat(site_pos[worldid % site_pos.shape[0], siteid], xquat)       <L 225>
        var_11 = &(var_site_pos.shape);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        var_15 = wp::mod(var_0, var_13);
        var_16 = wp::address(var_site_pos, var_15, var_1);
        var_18 = wp::load(var_16);
        var_17 = rot_vec_quat_0(var_18, var_9);
        var_19 = wp::add(var_6, var_17);
        wp::array_store(var_site_xpos_out, var_0, var_1, var_19);
        // site_xmat_out[worldid, siteid] = math.quat_to_mat(math.mul_quat(xquat, site_quat[worldid % site_quat.shape[0], siteid]))       <L 226>
        var_20 = &(var_site_quat.shape);
        var_23 = wp::load(var_20);
        var_22 = wp::extract(var_23, var_21);
        var_24 = wp::mod(var_0, var_22);
        var_25 = wp::address(var_site_quat, var_24, var_1);
        var_27 = wp::load(var_25);
        var_26 = mul_quat_0(var_9, var_27);
        var_28 = quat_to_mat_0(var_26);
        wp::array_store(var_site_xmat_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _tendon_armature_0d777ffa_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_M_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        const wp::float32 var_28 = 0.0;
        bool var_29;
        wp::int32* var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        bool var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        wp::int32 var_45;
        wp::int32* var_46;
        bool var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        bool var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        wp::int32* var_55;
        bool var_56;
        wp::int32 var_57;
        wp::float32* var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::float32 var_61 = 0.0;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::array_t<wp::float32> var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        //---------
        // forward
        // def _tendon_armature(                                                                  <L 1102>
        // worldid, tenid, dofid = wp.tid()                                                       <L 1116>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 1118>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 1120>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 1121>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 1123>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid >= rownnz:                                                                    <L 1124>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 1125>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 1126>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // dofid_sparse = dofid                                                                   <L 1127>
        var_20 = wp::copy(var_2);
        // sparseid = rowadr + dofid_sparse                                                       <L 1128>
        var_21 = wp::add(var_18, var_20);
        // dofid = ten_J_colind[sparseid]                                                         <L 1129>
        var_22 = wp::address(var_ten_J_colind, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // ten_Ji = ten_J_in[worldid, sparseid]                                                   <L 1130>
        var_25 = wp::address(var_ten_J_in, var_0, var_21);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // if ten_Ji == 0.0:                                                                      <L 1132>
        var_29 = (var_26 == var_28);
        if (var_29) {
            // return                                                                             <L 1133>
            continue;
        }
        // madr_ij = M_rowadr[dofid] + M_rownnz[dofid] - 1                                        <L 1136>
        var_30 = wp::address(var_M_rowadr, var_23);
        var_31 = wp::address(var_M_rownnz, var_23);
        var_33 = wp::load(var_30);
        var_34 = wp::load(var_31);
        var_32 = wp::add(var_33, var_34);
        var_36 = wp::sub(var_32, var_35);
        // dofidi = dofid                                                                         <L 1139>
        var_37 = wp::copy(var_23);
        // ptr = dofid_sparse                                                                     <L 1140>
        var_38 = wp::copy(var_20);
        // while dofid >= 0:                                                                      <L 1141>
        start_while_3:;
        var_40 = (var_23 >= var_39);
        if ((var_40) == false) goto end_while_3;
            // if dofid == dofidi:                                                                <L 1142>
            var_41 = (var_23 == var_37);
            if (var_41) {
                // ten_Jj = ten_Ji                                                                <L 1143>
                var_42 = wp::copy(var_26);
            }
            if (!var_41) {
                // while ptr >= 0:                                                                <L 1146>
        start_while_5:;
                var_44 = (var_38 >= var_43);
        if ((var_44) == false) goto end_while_5;
                    // sparseid = rowadr + ptr                                                    <L 1147>
                    var_45 = wp::add(var_18, var_38);
                    // if ten_J_colind[sparseid] <= dofid:                                        <L 1148>
                    var_46 = wp::address(var_ten_J_colind, var_45);
                    var_48 = wp::load(var_46);
                    var_47 = (var_48 <= var_23);
                    if (var_47) {
                        // break                                                                  <L 1149>
                        wp::assign(var_21, var_45);
                        goto end_while_5;
                    }
                    var_49 = wp::where(var_47, var_21, var_45);
                    // ptr -= 1                                                                   <L 1150>
                    var_51 = wp::sub(var_38, var_50);
                    wp::assign(var_21, var_49);
                    wp::assign(var_38, var_51);
        goto start_while_5;
        end_while_5:;
                // if ptr >= 0 and ten_J_colind[sparseid] == dofid:                               <L 1151>
                var_54 = (var_38 >= var_53);
                var_52 = var_54;
                if (var_52) {
                    var_55 = wp::address(var_ten_J_colind, var_21);
                    var_57 = wp::load(var_55);
                    var_56 = (var_57 == var_23);
                    var_52 = var_52 && var_56;
                }
                if (var_52) {
                    // ten_Jj = ten_J_in[worldid, sparseid]                                       <L 1152>
                    var_58 = wp::address(var_ten_J_in, var_0, var_21);
                    var_60 = wp::load(var_58);
                    var_59 = wp::copy(var_60);
                }
                if (!var_52) {
                    // ten_Jj = float(0.0)                                                        <L 1154>
                    var_62 = wp::float(var_61);
                }
                var_63 = wp::where(var_52, var_59, var_62);
            }
            var_64 = wp::where(var_41, var_42, var_63);
            // Mij = armature * ten_Jj * ten_Ji                                                   <L 1156>
            var_65 = wp::mul(var_9, var_64);
            var_66 = wp::mul(var_65, var_26);
            // wp.atomic_add(M_out[worldid], madr_ij, Mij)                                        <L 1158>
            var_67 = wp::view(var_M_out, var_0);
            var_68 = wp::atomic_add(var_67, var_36, var_66);
            // madr_ij -= 1                                                                       <L 1159>
            var_70 = wp::sub(var_36, var_69);
            // dofid = dof_parentid[dofid]                                                        <L 1161>
            var_71 = wp::address(var_dof_parentid, var_23);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            wp::assign(var_23, var_72);
            wp::assign(var_36, var_70);
        goto start_while_3;
        end_while_3:;
    }
}



extern "C" __global__ void _subtree_div_0ef57c26_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.0;
        bool var_14;
        wp::vec_t<3, wp::float32> var_15;
        //---------
        // forward
        // def _subtree_div(                                                                      <L 718>
        // worldid, bodyid = wp.tid()                                                             <L 726>
        builtin_tid2d(var_0, var_1);
        // com = subtree_com_in[worldid, bodyid]                                                  <L 727>
        var_2 = wp::address(var_subtree_com_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]                   <L 728>
        var_5 = &(var_body_subtreemass.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_subtreemass, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if mass != 0.0:                                                                        <L 729>
        var_14 = (var_11 != var_13);
        if (var_14) {
            // subtree_com_out[worldid, bodyid] = com / mass                                      <L 730>
            var_15 = wp::div(var_3, var_11);
            wp::array_store(var_subtree_com_out, var_0, var_1, var_15);
        }
    }
}



extern "C" __global__ void _comvel_branch_0f933c58_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::array_t<wp::float32> var_10;
        wp::array_t<wp::vec_t<6, wp::float32>> var_11;
        wp::range_t var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::vec_t<6, wp::float32>* var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::int32 var_34;
        wp::range_t var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        bool var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::vec_t<6, wp::float32>* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::vec_t<6, wp::float32> var_48;
        wp::vec_t<6, wp::float32> var_49;
        wp::float32 var_50;
        wp::vec_t<6, wp::float32> var_51;
        const wp::int32 var_52 = 1;
        wp::int32 var_53;
        wp::vec_t<6, wp::float32>* var_54;
        const wp::int32 var_55 = 1;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::vec_t<6, wp::float32> var_58;
        wp::vec_t<6, wp::float32> var_59;
        wp::float32 var_60;
        wp::vec_t<6, wp::float32> var_61;
        const wp::int32 var_62 = 2;
        wp::int32 var_63;
        wp::vec_t<6, wp::float32>* var_64;
        const wp::int32 var_65 = 2;
        wp::int32 var_66;
        wp::float32* var_67;
        wp::vec_t<6, wp::float32> var_68;
        wp::vec_t<6, wp::float32> var_69;
        wp::float32 var_70;
        wp::vec_t<6, wp::float32> var_71;
        const wp::float32 var_72 = 0.0;
        const wp::float32 var_73 = 0.0;
        const wp::float32 var_74 = 0.0;
        const wp::float32 var_75 = 0.0;
        const wp::float32 var_76 = 0.0;
        const wp::float32 var_77 = 0.0;
        wp::vec_t<6, wp::float32> var_78;
        const wp::int32 var_79 = 0;
        wp::int32 var_80;
        const wp::float32 var_81 = 0.0;
        const wp::float32 var_82 = 0.0;
        const wp::float32 var_83 = 0.0;
        const wp::float32 var_84 = 0.0;
        const wp::float32 var_85 = 0.0;
        const wp::float32 var_86 = 0.0;
        wp::vec_t<6, wp::float32> var_87;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        const wp::float32 var_90 = 0.0;
        const wp::float32 var_91 = 0.0;
        const wp::float32 var_92 = 0.0;
        const wp::float32 var_93 = 0.0;
        const wp::float32 var_94 = 0.0;
        const wp::float32 var_95 = 0.0;
        wp::vec_t<6, wp::float32> var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        const wp::int32 var_99 = 3;
        wp::int32 var_100;
        wp::vec_t<6, wp::float32>* var_101;
        wp::vec_t<6, wp::float32> var_102;
        wp::vec_t<6, wp::float32> var_103;
        const wp::int32 var_104 = 3;
        wp::int32 var_105;
        const wp::int32 var_106 = 4;
        wp::int32 var_107;
        wp::vec_t<6, wp::float32>* var_108;
        wp::vec_t<6, wp::float32> var_109;
        wp::vec_t<6, wp::float32> var_110;
        const wp::int32 var_111 = 4;
        wp::int32 var_112;
        const wp::int32 var_113 = 5;
        wp::int32 var_114;
        wp::vec_t<6, wp::float32>* var_115;
        wp::vec_t<6, wp::float32> var_116;
        wp::vec_t<6, wp::float32> var_117;
        const wp::int32 var_118 = 5;
        wp::int32 var_119;
        const wp::int32 var_120 = 3;
        wp::int32 var_121;
        wp::vec_t<6, wp::float32>* var_122;
        const wp::int32 var_123 = 3;
        wp::int32 var_124;
        wp::float32* var_125;
        wp::vec_t<6, wp::float32> var_126;
        wp::vec_t<6, wp::float32> var_127;
        wp::float32 var_128;
        wp::vec_t<6, wp::float32> var_129;
        const wp::int32 var_130 = 4;
        wp::int32 var_131;
        wp::vec_t<6, wp::float32>* var_132;
        const wp::int32 var_133 = 4;
        wp::int32 var_134;
        wp::float32* var_135;
        wp::vec_t<6, wp::float32> var_136;
        wp::vec_t<6, wp::float32> var_137;
        wp::float32 var_138;
        wp::vec_t<6, wp::float32> var_139;
        const wp::int32 var_140 = 5;
        wp::int32 var_141;
        wp::vec_t<6, wp::float32>* var_142;
        const wp::int32 var_143 = 5;
        wp::int32 var_144;
        wp::float32* var_145;
        wp::vec_t<6, wp::float32> var_146;
        wp::vec_t<6, wp::float32> var_147;
        wp::float32 var_148;
        wp::vec_t<6, wp::float32> var_149;
        const wp::int32 var_150 = 6;
        wp::int32 var_151;
        const wp::int32 var_152 = 1;
        bool var_153;
        const wp::int32 var_154 = 0;
        wp::int32 var_155;
        wp::vec_t<6, wp::float32>* var_156;
        wp::vec_t<6, wp::float32> var_157;
        wp::vec_t<6, wp::float32> var_158;
        const wp::int32 var_159 = 0;
        wp::int32 var_160;
        const wp::int32 var_161 = 1;
        wp::int32 var_162;
        wp::vec_t<6, wp::float32>* var_163;
        wp::vec_t<6, wp::float32> var_164;
        wp::vec_t<6, wp::float32> var_165;
        const wp::int32 var_166 = 1;
        wp::int32 var_167;
        const wp::int32 var_168 = 2;
        wp::int32 var_169;
        wp::vec_t<6, wp::float32>* var_170;
        wp::vec_t<6, wp::float32> var_171;
        wp::vec_t<6, wp::float32> var_172;
        const wp::int32 var_173 = 2;
        wp::int32 var_174;
        const wp::int32 var_175 = 0;
        wp::int32 var_176;
        wp::vec_t<6, wp::float32>* var_177;
        const wp::int32 var_178 = 0;
        wp::int32 var_179;
        wp::float32* var_180;
        wp::vec_t<6, wp::float32> var_181;
        wp::vec_t<6, wp::float32> var_182;
        wp::float32 var_183;
        wp::vec_t<6, wp::float32> var_184;
        const wp::int32 var_185 = 1;
        wp::int32 var_186;
        wp::vec_t<6, wp::float32>* var_187;
        const wp::int32 var_188 = 1;
        wp::int32 var_189;
        wp::float32* var_190;
        wp::vec_t<6, wp::float32> var_191;
        wp::vec_t<6, wp::float32> var_192;
        wp::float32 var_193;
        wp::vec_t<6, wp::float32> var_194;
        const wp::int32 var_195 = 2;
        wp::int32 var_196;
        wp::vec_t<6, wp::float32>* var_197;
        const wp::int32 var_198 = 2;
        wp::int32 var_199;
        wp::float32* var_200;
        wp::vec_t<6, wp::float32> var_201;
        wp::vec_t<6, wp::float32> var_202;
        wp::float32 var_203;
        wp::vec_t<6, wp::float32> var_204;
        const wp::int32 var_205 = 3;
        wp::int32 var_206;
        wp::vec_t<6, wp::float32>* var_207;
        wp::vec_t<6, wp::float32> var_208;
        wp::vec_t<6, wp::float32> var_209;
        wp::vec_t<6, wp::float32>* var_210;
        wp::float32* var_211;
        wp::vec_t<6, wp::float32> var_212;
        wp::vec_t<6, wp::float32> var_213;
        wp::float32 var_214;
        wp::vec_t<6, wp::float32> var_215;
        const wp::int32 var_216 = 1;
        wp::int32 var_217;
        wp::vec_t<6, wp::float32> var_218;
        wp::int32 var_219;
        wp::vec_t<6, wp::float32> var_220;
        wp::int32 var_221;
        //---------
        // forward
        // def _comvel_branch(                                                                    <L 2183>
        // worldid, branchid = wp.tid()                                                           <L 2199>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 2201>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 2202>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qvel = qvel_in[worldid]                                                                <L 2204>
        var_10 = wp::view(var_qvel_in, var_0);
        // cdof = cdof_in[worldid]                                                                <L 2205>
        var_11 = wp::view(var_cdof_in, var_0);
        // for i in range(start, end):                                                            <L 2207>
        var_12 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_12) == 0) goto end_for_0;
            var_13 = wp::iter_next(var_12);
            // bodyid = body_branches[i]                                                          <L 2208>
            var_14 = wp::address(var_body_branches, var_13);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // pid = body_parentid[bodyid]                                                        <L 2209>
            var_17 = wp::address(var_body_parentid, var_15);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // cvel = cvel_out[worldid, pid]                                                      <L 2210>
            var_20 = wp::address(var_cvel_out, var_0, var_18);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // dofid = body_dofadr[bodyid]                                                        <L 2211>
            var_23 = wp::address(var_body_dofadr, var_15);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // jntid = body_jntadr[bodyid]                                                        <L 2212>
            var_26 = wp::address(var_body_jntadr, var_15);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // jntnum = body_jntnum[bodyid]                                                       <L 2213>
            var_29 = wp::address(var_body_jntnum, var_15);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // if jntnum == 0:                                                                    <L 2215>
            var_33 = (var_30 == var_32);
            if (var_33) {
                // cvel_out[worldid, bodyid] = cvel                                               <L 2216>
                wp::array_store(var_cvel_out, var_0, var_15, var_21);
                // continue                                                                       <L 2217>
                goto start_for_0;
            }
            // for j in range(jntid, jntid + jntnum):                                             <L 2219>
            var_34 = wp::add(var_27, var_30);
            var_35 = wp::range(var_27, var_34);
            start_for_2:;
                if (iter_cmp(var_35) == 0) goto end_for_2;
                var_36 = wp::iter_next(var_35);
                // jnttype = jnt_type[j]                                                          <L 2220>
                var_37 = wp::address(var_jnt_type, var_36);
                var_39 = wp::load(var_37);
                var_38 = wp::copy(var_39);
                // if jnttype == JointType.FREE:                                                  <L 2222>
                var_41 = (var_38 == var_40);
                if (var_41) {
                    // cvel += cdof[dofid + 0] * qvel[dofid + 0]                                  <L 2223>
                    var_43 = wp::add(var_24, var_42);
                    var_44 = wp::address(var_11, var_43);
                    var_46 = wp::add(var_24, var_45);
                    var_47 = wp::address(var_10, var_46);
                    var_49 = wp::load(var_44);
                    var_50 = wp::load(var_47);
                    var_48 = wp::mul(var_49, var_50);
                    var_51 = wp::add(var_21, var_48);
                    // cvel += cdof[dofid + 1] * qvel[dofid + 1]                                  <L 2224>
                    var_53 = wp::add(var_24, var_52);
                    var_54 = wp::address(var_11, var_53);
                    var_56 = wp::add(var_24, var_55);
                    var_57 = wp::address(var_10, var_56);
                    var_59 = wp::load(var_54);
                    var_60 = wp::load(var_57);
                    var_58 = wp::mul(var_59, var_60);
                    var_61 = wp::add(var_51, var_58);
                    // cvel += cdof[dofid + 2] * qvel[dofid + 2]                                  <L 2225>
                    var_63 = wp::add(var_24, var_62);
                    var_64 = wp::address(var_11, var_63);
                    var_66 = wp::add(var_24, var_65);
                    var_67 = wp::address(var_10, var_66);
                    var_69 = wp::load(var_64);
                    var_70 = wp::load(var_67);
                    var_68 = wp::mul(var_69, var_70);
                    var_71 = wp::add(var_61, var_68);
                    // cdof_dot_out[worldid, dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 2227>
                    var_78 = wp::vec_t<6, wp::float32>({var_72, var_73, var_74, var_75, var_76, var_77});
                    var_80 = wp::add(var_24, var_79);
                    wp::array_store(var_cdof_dot_out, var_0, var_80, var_78);
                    // cdof_dot_out[worldid, dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 2228>
                    var_87 = wp::vec_t<6, wp::float32>({var_81, var_82, var_83, var_84, var_85, var_86});
                    var_89 = wp::add(var_24, var_88);
                    wp::array_store(var_cdof_dot_out, var_0, var_89, var_87);
                    // cdof_dot_out[worldid, dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 2229>
                    var_96 = wp::vec_t<6, wp::float32>({var_90, var_91, var_92, var_93, var_94, var_95});
                    var_98 = wp::add(var_24, var_97);
                    wp::array_store(var_cdof_dot_out, var_0, var_98, var_96);
                    // cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])       <L 2230>
                    var_100 = wp::add(var_24, var_99);
                    var_101 = wp::address(var_11, var_100);
                    var_103 = wp::load(var_101);
                    var_102 = motion_cross_0(var_71, var_103);
                    var_105 = wp::add(var_24, var_104);
                    wp::array_store(var_cdof_dot_out, var_0, var_105, var_102);
                    // cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])       <L 2231>
                    var_107 = wp::add(var_24, var_106);
                    var_108 = wp::address(var_11, var_107);
                    var_110 = wp::load(var_108);
                    var_109 = motion_cross_0(var_71, var_110);
                    var_112 = wp::add(var_24, var_111);
                    wp::array_store(var_cdof_dot_out, var_0, var_112, var_109);
                    // cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])       <L 2232>
                    var_114 = wp::add(var_24, var_113);
                    var_115 = wp::address(var_11, var_114);
                    var_117 = wp::load(var_115);
                    var_116 = motion_cross_0(var_71, var_117);
                    var_119 = wp::add(var_24, var_118);
                    wp::array_store(var_cdof_dot_out, var_0, var_119, var_116);
                    // cvel += cdof[dofid + 3] * qvel[dofid + 3]                                  <L 2234>
                    var_121 = wp::add(var_24, var_120);
                    var_122 = wp::address(var_11, var_121);
                    var_124 = wp::add(var_24, var_123);
                    var_125 = wp::address(var_10, var_124);
                    var_127 = wp::load(var_122);
                    var_128 = wp::load(var_125);
                    var_126 = wp::mul(var_127, var_128);
                    var_129 = wp::add(var_71, var_126);
                    // cvel += cdof[dofid + 4] * qvel[dofid + 4]                                  <L 2235>
                    var_131 = wp::add(var_24, var_130);
                    var_132 = wp::address(var_11, var_131);
                    var_134 = wp::add(var_24, var_133);
                    var_135 = wp::address(var_10, var_134);
                    var_137 = wp::load(var_132);
                    var_138 = wp::load(var_135);
                    var_136 = wp::mul(var_137, var_138);
                    var_139 = wp::add(var_129, var_136);
                    // cvel += cdof[dofid + 5] * qvel[dofid + 5]                                  <L 2236>
                    var_141 = wp::add(var_24, var_140);
                    var_142 = wp::address(var_11, var_141);
                    var_144 = wp::add(var_24, var_143);
                    var_145 = wp::address(var_10, var_144);
                    var_147 = wp::load(var_142);
                    var_148 = wp::load(var_145);
                    var_146 = wp::mul(var_147, var_148);
                    var_149 = wp::add(var_139, var_146);
                    // dofid += 6                                                                 <L 2238>
                    var_151 = wp::add(var_24, var_150);
                }
                if (!var_41) {
                    // elif jnttype == JointType.BALL:                                            <L 2239>
                    var_153 = (var_38 == var_152);
                    if (var_153) {
                        // cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])       <L 2240>
                        var_155 = wp::add(var_24, var_154);
                        var_156 = wp::address(var_11, var_155);
                        var_158 = wp::load(var_156);
                        var_157 = motion_cross_0(var_21, var_158);
                        var_160 = wp::add(var_24, var_159);
                        wp::array_store(var_cdof_dot_out, var_0, var_160, var_157);
                        // cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])       <L 2241>
                        var_162 = wp::add(var_24, var_161);
                        var_163 = wp::address(var_11, var_162);
                        var_165 = wp::load(var_163);
                        var_164 = motion_cross_0(var_21, var_165);
                        var_167 = wp::add(var_24, var_166);
                        wp::array_store(var_cdof_dot_out, var_0, var_167, var_164);
                        // cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])       <L 2242>
                        var_169 = wp::add(var_24, var_168);
                        var_170 = wp::address(var_11, var_169);
                        var_172 = wp::load(var_170);
                        var_171 = motion_cross_0(var_21, var_172);
                        var_174 = wp::add(var_24, var_173);
                        wp::array_store(var_cdof_dot_out, var_0, var_174, var_171);
                        // cvel += cdof[dofid + 0] * qvel[dofid + 0]                              <L 2244>
                        var_176 = wp::add(var_24, var_175);
                        var_177 = wp::address(var_11, var_176);
                        var_179 = wp::add(var_24, var_178);
                        var_180 = wp::address(var_10, var_179);
                        var_182 = wp::load(var_177);
                        var_183 = wp::load(var_180);
                        var_181 = wp::mul(var_182, var_183);
                        var_184 = wp::add(var_21, var_181);
                        // cvel += cdof[dofid + 1] * qvel[dofid + 1]                              <L 2245>
                        var_186 = wp::add(var_24, var_185);
                        var_187 = wp::address(var_11, var_186);
                        var_189 = wp::add(var_24, var_188);
                        var_190 = wp::address(var_10, var_189);
                        var_192 = wp::load(var_187);
                        var_193 = wp::load(var_190);
                        var_191 = wp::mul(var_192, var_193);
                        var_194 = wp::add(var_184, var_191);
                        // cvel += cdof[dofid + 2] * qvel[dofid + 2]                              <L 2246>
                        var_196 = wp::add(var_24, var_195);
                        var_197 = wp::address(var_11, var_196);
                        var_199 = wp::add(var_24, var_198);
                        var_200 = wp::address(var_10, var_199);
                        var_202 = wp::load(var_197);
                        var_203 = wp::load(var_200);
                        var_201 = wp::mul(var_202, var_203);
                        var_204 = wp::add(var_194, var_201);
                        // dofid += 3                                                             <L 2248>
                        var_206 = wp::add(var_24, var_205);
                    }
                    if (!var_153) {
                        // cdof_dot_out[worldid, dofid] = math.motion_cross(cvel, cdof[dofid])       <L 2250>
                        var_207 = wp::address(var_11, var_24);
                        var_209 = wp::load(var_207);
                        var_208 = motion_cross_0(var_21, var_209);
                        wp::array_store(var_cdof_dot_out, var_0, var_24, var_208);
                        // cvel += cdof[dofid] * qvel[dofid]                                      <L 2251>
                        var_210 = wp::address(var_11, var_24);
                        var_211 = wp::address(var_10, var_24);
                        var_213 = wp::load(var_210);
                        var_214 = wp::load(var_211);
                        var_212 = wp::mul(var_213, var_214);
                        var_215 = wp::add(var_21, var_212);
                        // dofid += 1                                                             <L 2253>
                        var_217 = wp::add(var_24, var_216);
                    }
                    var_218 = wp::where(var_153, var_204, var_215);
                    var_219 = wp::where(var_153, var_206, var_217);
                }
                var_220 = wp::where(var_41, var_149, var_218);
                var_221 = wp::where(var_41, var_151, var_219);
                wp::assign(var_21, var_220);
                wp::assign(var_24, var_221);
                goto start_for_2;
            end_for_2:;
            // cvel_out[worldid, bodyid] = cvel                                                   <L 2255>
            wp::array_store(var_cvel_out, var_0, var_15, var_21);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _qLD_acc_120d68f2_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::vec_t<3, wp::int32>> var_qLD_updates_,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_L_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::int32>* var_2;
        wp::vec_t<3, wp::int32> var_3;
        wp::vec_t<3, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::int32* var_26;
        wp::range_t var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::array_t<wp::float32> var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        //---------
        // forward
        // def _qLD_acc(                                                                          <L 1185>
        // worldid, nodeid = wp.tid()                                                             <L 1195>
        builtin_tid2d(var_0, var_1);
        // update = qLD_updates_[nodeid]                                                          <L 1196>
        var_2 = wp::address(var_qLD_updates_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // i, k, Madr_ki = update[0], update[1], update[2]                                        <L 1197>
        var_6 = wp::extract(var_3, var_5);
        var_8 = wp::extract(var_3, var_7);
        var_10 = wp::extract(var_3, var_9);
        // Madr_i = M_rowadr[i]  # Address of row being updated                                   <L 1198>
        var_11 = wp::address(var_M_rowadr, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // diag_k = M_rowadr[k] + M_rownnz[k] - 1  # Address of diagonal element of k             <L 1199>
        var_14 = wp::address(var_M_rowadr, var_8);
        var_15 = wp::address(var_M_rownnz, var_8);
        var_17 = wp::load(var_14);
        var_18 = wp::load(var_15);
        var_16 = wp::add(var_17, var_18);
        var_20 = wp::sub(var_16, var_19);
        // tmp = L_out[worldid, Madr_ki] / L_out[worldid, diag_k]                                 <L 1201>
        var_21 = wp::address(var_L_out, var_0, var_10);
        var_22 = wp::address(var_L_out, var_0, var_20);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = wp::div(var_24, var_25);
        // for j in range(M_rownnz[i]):                                                           <L 1202>
        var_26 = wp::address(var_M_rownnz, var_6);
        var_28 = wp::load(var_26);
        var_27 = wp::range(var_28);
        start_for_0:;
            if (iter_cmp(var_27) == 0) goto end_for_0;
            var_29 = wp::iter_next(var_27);
            // wp.atomic_sub(L_out[worldid], Madr_i + j, L_in[worldid, M_rowadr[k] + j] * tmp)       <L 1204>
            var_30 = wp::view(var_L_out, var_0);
            var_31 = wp::add(var_12, var_29);
            var_32 = wp::address(var_M_rowadr, var_8);
            var_34 = wp::load(var_32);
            var_33 = wp::add(var_34, var_29);
            var_35 = wp::address(var_L_in, var_0, var_33);
            var_37 = wp::load(var_35);
            var_36 = wp::mul(var_37, var_23);
            var_38 = wp::atomic_sub(var_30, var_31, var_36);
            goto start_for_0;
        end_for_0:;
        // L_out[worldid, Madr_ki] = tmp                                                          <L 1206>
        wp::array_store(var_L_out, var_0, var_10, var_23);
    }
}



extern "C" __global__ void _cinert_1261260c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::mat_t<3, 3, wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::mat_t<3, 3, wp::float32> var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32* var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<10, wp::float32> var_28;
        wp::mat_t<3, 3, wp::float32> var_29;
        wp::mat_t<3, 3, wp::float32> var_30;
        wp::mat_t<3, 3, wp::float32> var_31;
        wp::mat_t<3, 3, wp::float32> var_32;
        const wp::int32 var_33 = 0;
        const wp::int32 var_34 = 0;
        wp::float32 var_35;
        const wp::int32 var_36 = 0;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        const wp::int32 var_41 = 2;
        const wp::int32 var_42 = 2;
        wp::float32 var_43;
        const wp::int32 var_44 = 2;
        const wp::int32 var_45 = 0;
        const wp::int32 var_46 = 1;
        wp::float32 var_47;
        const wp::int32 var_48 = 3;
        const wp::int32 var_49 = 0;
        const wp::int32 var_50 = 2;
        wp::float32 var_51;
        const wp::int32 var_52 = 4;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        const wp::int32 var_56 = 5;
        const wp::int32 var_57 = 1;
        wp::float32 var_58;
        const wp::int32 var_59 = 1;
        wp::float32 var_60;
        wp::float32 var_61;
        const wp::int32 var_62 = 2;
        wp::float32 var_63;
        const wp::int32 var_64 = 2;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 0;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        const wp::int32 var_72 = 0;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 2;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 1;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        const wp::int32 var_85 = 0;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 1;
        wp::float32 var_89;
        const wp::int32 var_90 = 1;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        const wp::int32 var_95 = 2;
        const wp::int32 var_96 = 0;
        wp::float32 var_97;
        wp::float32 var_98;
        const wp::int32 var_99 = 1;
        wp::float32 var_100;
        wp::float32 var_101;
        const wp::int32 var_102 = 3;
        const wp::int32 var_103 = 0;
        wp::float32 var_104;
        wp::float32 var_105;
        const wp::int32 var_106 = 2;
        wp::float32 var_107;
        wp::float32 var_108;
        const wp::int32 var_109 = 4;
        const wp::int32 var_110 = 1;
        wp::float32 var_111;
        wp::float32 var_112;
        const wp::int32 var_113 = 2;
        wp::float32 var_114;
        wp::float32 var_115;
        const wp::int32 var_116 = 5;
        const wp::int32 var_117 = 0;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 6;
        const wp::int32 var_121 = 1;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 7;
        const wp::int32 var_125 = 2;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 8;
        const wp::int32 var_129 = 9;
        //---------
        // forward
        // def _cinert(                                                                           <L 734>
        // worldid, bodyid = wp.tid()                                                             <L 746>
        builtin_tid2d(var_0, var_1);
        // mat = ximat_in[worldid, bodyid]                                                        <L 747>
        var_2 = wp::address(var_ximat_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // inert = body_inertia[worldid % body_inertia.shape[0], bodyid]                          <L 748>
        var_5 = &(var_body_inertia.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_inertia, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 749>
        var_13 = &(var_body_mass.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_body_mass, var_17, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // dif = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]         <L 750>
        var_21 = wp::address(var_xipos_in, var_0, var_1);
        var_22 = wp::address(var_body_rootid, var_1);
        var_24 = wp::load(var_22);
        var_23 = wp::address(var_subtree_com_in, var_0, var_24);
        var_26 = wp::load(var_21);
        var_27 = wp::load(var_23);
        var_25 = wp::sub(var_26, var_27);
        // res = vec10()                                                                          <L 753>
        var_28 = wp::vec_t<10, wp::float32>();
        // tmp = mat @ wp.diag(inert) @ wp.transpose(mat)                                         <L 755>
        var_29 = wp::diag(var_11);
        var_30 = wp::mul(var_3, var_29);
        var_31 = wp::transpose(var_3);
        var_32 = wp::mul(var_30, var_31);
        // res[0] = tmp[0, 0]                                                                     <L 756>
        var_35 = wp::extract(var_32, var_33, var_34);
        wp::assign_inplace(var_28, var_36, var_35);
        // res[1] = tmp[1, 1]                                                                     <L 757>
        var_39 = wp::extract(var_32, var_37, var_38);
        wp::assign_inplace(var_28, var_40, var_39);
        // res[2] = tmp[2, 2]                                                                     <L 758>
        var_43 = wp::extract(var_32, var_41, var_42);
        wp::assign_inplace(var_28, var_44, var_43);
        // res[3] = tmp[0, 1]                                                                     <L 759>
        var_47 = wp::extract(var_32, var_45, var_46);
        wp::assign_inplace(var_28, var_48, var_47);
        // res[4] = tmp[0, 2]                                                                     <L 760>
        var_51 = wp::extract(var_32, var_49, var_50);
        wp::assign_inplace(var_28, var_52, var_51);
        // res[5] = tmp[1, 2]                                                                     <L 761>
        var_55 = wp::extract(var_32, var_53, var_54);
        wp::assign_inplace(var_28, var_56, var_55);
        // res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])                                   <L 763>
        var_58 = wp::extract(var_25, var_57);
        var_60 = wp::extract(var_25, var_59);
        var_61 = wp::mul(var_58, var_60);
        var_63 = wp::extract(var_25, var_62);
        var_65 = wp::extract(var_25, var_64);
        var_66 = wp::mul(var_63, var_65);
        var_67 = wp::add(var_61, var_66);
        var_68 = wp::mul(var_19, var_67);
        wp::add_inplace(var_28, var_69, var_68);
        // res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])                                   <L 764>
        var_71 = wp::extract(var_25, var_70);
        var_73 = wp::extract(var_25, var_72);
        var_74 = wp::mul(var_71, var_73);
        var_76 = wp::extract(var_25, var_75);
        var_78 = wp::extract(var_25, var_77);
        var_79 = wp::mul(var_76, var_78);
        var_80 = wp::add(var_74, var_79);
        var_81 = wp::mul(var_19, var_80);
        wp::add_inplace(var_28, var_82, var_81);
        // res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])                                   <L 765>
        var_84 = wp::extract(var_25, var_83);
        var_86 = wp::extract(var_25, var_85);
        var_87 = wp::mul(var_84, var_86);
        var_89 = wp::extract(var_25, var_88);
        var_91 = wp::extract(var_25, var_90);
        var_92 = wp::mul(var_89, var_91);
        var_93 = wp::add(var_87, var_92);
        var_94 = wp::mul(var_19, var_93);
        wp::add_inplace(var_28, var_95, var_94);
        // res[3] -= mass * dif[0] * dif[1]                                                       <L 766>
        var_97 = wp::extract(var_25, var_96);
        var_98 = wp::mul(var_19, var_97);
        var_100 = wp::extract(var_25, var_99);
        var_101 = wp::mul(var_98, var_100);
        wp::sub_inplace(var_28, var_102, var_101);
        // res[4] -= mass * dif[0] * dif[2]                                                       <L 767>
        var_104 = wp::extract(var_25, var_103);
        var_105 = wp::mul(var_19, var_104);
        var_107 = wp::extract(var_25, var_106);
        var_108 = wp::mul(var_105, var_107);
        wp::sub_inplace(var_28, var_109, var_108);
        // res[5] -= mass * dif[1] * dif[2]                                                       <L 768>
        var_111 = wp::extract(var_25, var_110);
        var_112 = wp::mul(var_19, var_111);
        var_114 = wp::extract(var_25, var_113);
        var_115 = wp::mul(var_112, var_114);
        wp::sub_inplace(var_28, var_116, var_115);
        // res[6] = mass * dif[0]                                                                 <L 770>
        var_118 = wp::extract(var_25, var_117);
        var_119 = wp::mul(var_19, var_118);
        wp::assign_inplace(var_28, var_120, var_119);
        // res[7] = mass * dif[1]                                                                 <L 771>
        var_122 = wp::extract(var_25, var_121);
        var_123 = wp::mul(var_19, var_122);
        wp::assign_inplace(var_28, var_124, var_123);
        // res[8] = mass * dif[2]                                                                 <L 772>
        var_126 = wp::extract(var_25, var_125);
        var_127 = wp::mul(var_19, var_126);
        wp::assign_inplace(var_28, var_128, var_127);
        // res[9] = mass                                                                          <L 774>
        wp::assign_inplace(var_28, var_129, var_19);
        // cinert_out[worldid, bodyid] = res                                                      <L 776>
        wp::array_store(var_cinert_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _M_13671fca_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::float32> var_dof_armature,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_M_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::vec_t<10, wp::float32>* var_19;
        wp::vec_t<6, wp::float32>* var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<10, wp::float32> var_22;
        wp::vec_t<6, wp::float32> var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::vec_t<6, wp::float32>* var_26;
        wp::float32 var_27;
        wp::vec_t<6, wp::float32> var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        //---------
        // forward
        // def _M(                                                                                <L 1049>
        // worldid, dofid = wp.tid()                                                              <L 1062>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 1063>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // madr_ij = M_rowadr[dofid] + M_rownnz[dofid] - 1                                        <L 1064>
        var_5 = wp::address(var_M_rowadr, var_1);
        var_6 = wp::address(var_M_rownnz, var_1);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::add(var_8, var_9);
        var_11 = wp::sub(var_7, var_10);
        // M_out[worldid, madr_ij] = dof_armature[worldid % dof_armature.shape[0], dofid]         <L 1067>
        var_12 = &(var_dof_armature.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        var_17 = wp::address(var_dof_armature, var_16, var_1);
        var_18 = wp::load(var_17);
        wp::array_store(var_M_out, var_0, var_11, var_18);
        // buf = math.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])                 <L 1070>
        var_19 = wp::address(var_crb_in, var_0, var_3);
        var_20 = wp::address(var_cdof_in, var_0, var_1);
        var_22 = wp::load(var_19);
        var_23 = wp::load(var_20);
        var_21 = inert_vec_0(var_22, var_23);
        // while dofid >= 0:                                                                      <L 1073>
        start_while_0:;
        var_25 = (var_1 >= var_24);
        if ((var_25) == false) goto end_while_0;
            // M_out[worldid, madr_ij] += wp.dot(cdof_in[worldid, dofid], buf)                    <L 1074>
            var_26 = wp::address(var_cdof_in, var_0, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::dot(var_28, var_21);
            var_29 = wp::atomic_add(var_M_out, var_0, var_11, var_27);
            // madr_ij -= 1                                                                       <L 1075>
            var_31 = wp::sub(var_11, var_30);
            // dofid = dof_parentid[dofid]                                                        <L 1076>
            var_32 = wp::address(var_dof_parentid, var_1);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            wp::assign(var_1, var_33);
            wp::assign(var_11, var_31);
        goto start_while_0;
        end_while_0:;
    }
}



extern "C" __global__ void _comvel_root_15820f35_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::float32 var_2 = 0.0;
        const wp::int32 var_3 = 0;
        //---------
        // forward
        // def _comvel_root(cvel_out: wp.array2d[wp.spatial_vector]):                             <L 2177>
        // worldid, elementid = wp.tid()                                                          <L 2178>
        builtin_tid2d(var_0, var_1);
        // cvel_out[worldid, 0][elementid] = 0.0                                                  <L 2179>
        wp::index(var_cvel_out, var_0, var_3)[var_1] = var_2;
    }
}



extern "C" __global__ void _light_local_to_global_1966b84b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_light_mode,
    wp::array_t<wp::int32> var_light_bodyid,
    wp::array_t<wp::int32> var_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_poscom0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32* var_13;
        const wp::int32 var_14 = 3;
        bool var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 4;
        bool var_19;
        wp::int32 var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::quat_t<wp::float32>* var_32;
        wp::quat_t<wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::int32* var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        wp::int32 var_45;
        wp::shape_t* var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        wp::shape_t var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::int32* var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::int32 var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::shape_t* var_58;
        const wp::int32 var_59 = 0;
        wp::int32 var_60;
        wp::shape_t var_61;
        wp::int32 var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::int32* var_66;
        const wp::int32 var_67 = 2;
        bool var_68;
        wp::int32 var_69;
        wp::shape_t* var_70;
        const wp::int32 var_71 = 0;
        wp::int32 var_72;
        wp::shape_t var_73;
        wp::int32 var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::int32* var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::int32 var_79;
        wp::shape_t* var_80;
        const wp::int32 var_81 = 0;
        wp::int32 var_82;
        wp::shape_t var_83;
        wp::int32 var_84;
        wp::vec_t<3, wp::float32>* var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        bool var_89;
        wp::int32* var_90;
        const wp::int32 var_91 = 3;
        bool var_92;
        wp::int32 var_93;
        wp::int32* var_94;
        const wp::int32 var_95 = 4;
        bool var_96;
        wp::int32 var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::vec_t<3, wp::float32>* var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::quat_t<wp::float32>* var_104;
        wp::quat_t<wp::float32> var_105;
        wp::quat_t<wp::float32> var_106;
        wp::vec_t<3, wp::float32>* var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::int32* var_111;
        wp::vec_t<3, wp::float32>* var_112;
        wp::int32 var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::int32* var_116;
        const wp::int32 var_117 = 4;
        bool var_118;
        wp::int32 var_119;
        wp::int32* var_120;
        wp::vec_t<3, wp::float32>* var_121;
        wp::int32 var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32>* var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::int32* var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::vec_t<3, wp::float32>* var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::vec_t<3, wp::float32> var_134;
        wp::quat_t<wp::float32>* var_135;
        wp::quat_t<wp::float32> var_136;
        wp::quat_t<wp::float32> var_137;
        wp::vec_t<3, wp::float32>* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32>* var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::int32 var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::quat_t<wp::float32> var_147;
        wp::int32 var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::quat_t<wp::float32> var_150;
        wp::vec_t<3, wp::float32>* var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        //---------
        // forward
        // def _light_local_to_global(                                                            <L 926>
        // worldid, lightid = wp.tid()                                                            <L 944>
        builtin_tid2d(var_0, var_1);
        // light_pos_id = worldid % light_pos.shape[0]                                            <L 945>
        var_2 = &(var_light_pos.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // light_dir_id = worldid % light_dir.shape[0]                                            <L 946>
        var_7 = &(var_light_dir.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // is_target_light = (light_mode[lightid] == CamLightType.TARGETBODY) or (light_mode[lightid] == CamLightType.TARGETBODYCOM)       <L 947>
        var_13 = wp::address(var_light_mode, var_1);
        var_16 = wp::load(var_13);
        var_15 = (var_16 == var_14);
        var_12 = var_15;
        if (!var_12) {
            var_17 = wp::address(var_light_mode, var_1);
            var_20 = wp::load(var_17);
            var_19 = (var_20 == var_18);
            var_12 = var_12 || var_19;
        }
        // invalid_target = is_target_light and (light_targetbodyid[lightid] < 0)                 <L 948>
        var_21 = var_12;
        if (var_21) {
            var_22 = wp::address(var_light_targetbodyid, var_1);
            var_25 = wp::load(var_22);
            var_24 = (var_25 < var_23);
            var_21 = var_21 && var_24;
        }
        // if invalid_target:                                                                     <L 949>
        if (var_21) {
            // bodyid = light_bodyid[lightid]                                                     <L 950>
            var_26 = wp::address(var_light_bodyid, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 951>
            var_29 = wp::address(var_xpos_in, var_0, var_27);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xquat = xquat_in[worldid, bodyid]                                                  <L 952>
            var_32 = wp::address(var_xquat_in, var_0, var_27);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 953>
            var_35 = wp::address(var_light_pos, var_6, var_1);
            var_37 = wp::load(var_35);
            var_36 = rot_vec_quat_0(var_37, var_33);
            var_38 = wp::add(var_30, var_36);
            wp::array_store(var_light_xpos_out, var_0, var_1, var_38);
            // light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)       <L 954>
            var_39 = wp::address(var_light_dir, var_11, var_1);
            var_41 = wp::load(var_39);
            var_40 = rot_vec_quat_0(var_41, var_33);
            wp::array_store(var_light_xdir_out, var_0, var_1, var_40);
            // return                                                                             <L 955>
            continue;
        }
        if (!var_21) {
            // elif light_mode[lightid] == CamLightType.TRACK:                                    <L 956>
            var_42 = wp::address(var_light_mode, var_1);
            var_45 = wp::load(var_42);
            var_44 = (var_45 == var_43);
            if (var_44) {
                // light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]       <L 957>
                var_46 = &(var_light_dir0.shape);
                var_49 = wp::load(var_46);
                var_48 = wp::extract(var_49, var_47);
                var_50 = wp::mod(var_0, var_48);
                var_51 = wp::address(var_light_dir0, var_50, var_1);
                var_52 = wp::load(var_51);
                wp::array_store(var_light_xdir_out, var_0, var_1, var_52);
                // body_xpos = xpos_in[worldid, light_bodyid[lightid]]                            <L 958>
                var_53 = wp::address(var_light_bodyid, var_1);
                var_55 = wp::load(var_53);
                var_54 = wp::address(var_xpos_in, var_0, var_55);
                var_57 = wp::load(var_54);
                var_56 = wp::copy(var_57);
                // light_xpos_out[worldid, lightid] = body_xpos + light_pos0[worldid % light_pos0.shape[0], lightid]       <L 959>
                var_58 = &(var_light_pos0.shape);
                var_61 = wp::load(var_58);
                var_60 = wp::extract(var_61, var_59);
                var_62 = wp::mod(var_0, var_60);
                var_63 = wp::address(var_light_pos0, var_62, var_1);
                var_65 = wp::load(var_63);
                var_64 = wp::add(var_56, var_65);
                wp::array_store(var_light_xpos_out, var_0, var_1, var_64);
            }
            if (!var_44) {
                // elif light_mode[lightid] == CamLightType.TRACKCOM:                             <L 960>
                var_66 = wp::address(var_light_mode, var_1);
                var_69 = wp::load(var_66);
                var_68 = (var_69 == var_67);
                if (var_68) {
                    // light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]       <L 961>
                    var_70 = &(var_light_dir0.shape);
                    var_73 = wp::load(var_70);
                    var_72 = wp::extract(var_73, var_71);
                    var_74 = wp::mod(var_0, var_72);
                    var_75 = wp::address(var_light_dir0, var_74, var_1);
                    var_76 = wp::load(var_75);
                    wp::array_store(var_light_xdir_out, var_0, var_1, var_76);
                    // light_xpos_out[worldid, lightid] = (                                       <L 962>
                    // subtree_com_in[worldid, light_bodyid[lightid]] + light_poscom0[worldid % light_poscom0.shape[0], lightid]       <L 963>
                    var_77 = wp::address(var_light_bodyid, var_1);
                    var_79 = wp::load(var_77);
                    var_78 = wp::address(var_subtree_com_in, var_0, var_79);
                    var_80 = &(var_light_poscom0.shape);
                    var_83 = wp::load(var_80);
                    var_82 = wp::extract(var_83, var_81);
                    var_84 = wp::mod(var_0, var_82);
                    var_85 = wp::address(var_light_poscom0, var_84, var_1);
                    var_87 = wp::load(var_78);
                    var_88 = wp::load(var_85);
                    var_86 = wp::add(var_87, var_88);
                    // light_xpos_out[worldid, lightid] = (                                       <L 962>
                    wp::array_store(var_light_xpos_out, var_0, var_1, var_86);
                }
                if (!var_68) {
                    // elif light_mode[lightid] == CamLightType.TARGETBODY or light_mode[lightid] == CamLightType.TARGETBODYCOM:       <L 965>
                    var_90 = wp::address(var_light_mode, var_1);
                    var_93 = wp::load(var_90);
                    var_92 = (var_93 == var_91);
                    var_89 = var_92;
                    if (!var_89) {
                        var_94 = wp::address(var_light_mode, var_1);
                        var_97 = wp::load(var_94);
                        var_96 = (var_97 == var_95);
                        var_89 = var_89 || var_96;
                    }
                    if (var_89) {
                        // bodyid = light_bodyid[lightid]                                         <L 966>
                        var_98 = wp::address(var_light_bodyid, var_1);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 967>
                        var_101 = wp::address(var_xpos_in, var_0, var_99);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 968>
                        var_104 = wp::address(var_xquat_in, var_0, var_99);
                        var_106 = wp::load(var_104);
                        var_105 = wp::copy(var_106);
                        // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 969>
                        var_107 = wp::address(var_light_pos, var_6, var_1);
                        var_109 = wp::load(var_107);
                        var_108 = rot_vec_quat_0(var_109, var_105);
                        var_110 = wp::add(var_102, var_108);
                        wp::array_store(var_light_xpos_out, var_0, var_1, var_110);
                        // pos = xpos_in[worldid, light_targetbodyid[lightid]]                    <L 970>
                        var_111 = wp::address(var_light_targetbodyid, var_1);
                        var_113 = wp::load(var_111);
                        var_112 = wp::address(var_xpos_in, var_0, var_113);
                        var_115 = wp::load(var_112);
                        var_114 = wp::copy(var_115);
                        // if light_mode[lightid] == CamLightType.TARGETBODYCOM:                  <L 971>
                        var_116 = wp::address(var_light_mode, var_1);
                        var_119 = wp::load(var_116);
                        var_118 = (var_119 == var_117);
                        if (var_118) {
                            // pos = subtree_com_in[worldid, light_targetbodyid[lightid]]         <L 972>
                            var_120 = wp::address(var_light_targetbodyid, var_1);
                            var_122 = wp::load(var_120);
                            var_121 = wp::address(var_subtree_com_in, var_0, var_122);
                            var_124 = wp::load(var_121);
                            var_123 = wp::copy(var_124);
                        }
                        var_125 = wp::where(var_118, var_123, var_114);
                        // light_xdir_out[worldid, lightid] = pos - light_xpos_out[worldid, lightid]       <L 973>
                        var_126 = wp::address(var_light_xpos_out, var_0, var_1);
                        var_128 = wp::load(var_126);
                        var_127 = wp::sub(var_125, var_128);
                        wp::array_store(var_light_xdir_out, var_0, var_1, var_127);
                    }
                    if (!var_89) {
                        // bodyid = light_bodyid[lightid]                                         <L 975>
                        var_129 = wp::address(var_light_bodyid, var_1);
                        var_131 = wp::load(var_129);
                        var_130 = wp::copy(var_131);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 976>
                        var_132 = wp::address(var_xpos_in, var_0, var_130);
                        var_134 = wp::load(var_132);
                        var_133 = wp::copy(var_134);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 977>
                        var_135 = wp::address(var_xquat_in, var_0, var_130);
                        var_137 = wp::load(var_135);
                        var_136 = wp::copy(var_137);
                        // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 978>
                        var_138 = wp::address(var_light_pos, var_6, var_1);
                        var_140 = wp::load(var_138);
                        var_139 = rot_vec_quat_0(var_140, var_136);
                        var_141 = wp::add(var_133, var_139);
                        wp::array_store(var_light_xpos_out, var_0, var_1, var_141);
                        // light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)       <L 979>
                        var_142 = wp::address(var_light_dir, var_11, var_1);
                        var_144 = wp::load(var_142);
                        var_143 = rot_vec_quat_0(var_144, var_136);
                        wp::array_store(var_light_xdir_out, var_0, var_1, var_143);
                    }
                    var_145 = wp::where(var_89, var_99, var_130);
                    var_146 = wp::where(var_89, var_102, var_133);
                    var_147 = wp::where(var_89, var_105, var_136);
                }
            }
        }
        var_148 = wp::where(var_21, var_27, var_145);
        var_149 = wp::where(var_21, var_30, var_146);
        var_150 = wp::where(var_21, var_33, var_147);
        // light_xdir_out[worldid, lightid] = wp.normalize(light_xdir_out[worldid, lightid])       <L 981>
        var_151 = wp::address(var_light_xdir_out, var_0, var_1);
        var_153 = wp::load(var_151);
        var_152 = wp::normalize(var_153);
        wp::array_store(var_light_xdir_out, var_0, var_1, var_152);
    }
}



extern "C" __global__ void _spatial_geom_tendon_1ca37386_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::int32> var_tendon_geom_adr,
    wp::array_t<wp::int32> var_wrap_geom_adr,
    wp::array_t<wp::float32> var_wrap_pulley_scale,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_geom_xpos_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::mat_t<3, 3, wp::float32>* var_35;
        wp::mat_t<3, 3, wp::float32> var_36;
        wp::mat_t<3, 3, wp::float32> var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32>* var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::float32 var_68 = 10000000000.0;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<6, wp::float32> var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::float32 var_81 = 0.0;
        bool var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::float32 var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::array_t<wp::float32> var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        const wp::float32 var_94 = 1e-15;
        bool var_95;
        const wp::float32 var_96 = 1.0;
        const wp::float32 var_97 = 0.0;
        const wp::float32 var_98 = 0.0;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        bool var_101;
        const wp::float32 var_102 = 1.0;
        const wp::float32 var_103 = 0.0;
        const wp::float32 var_104 = 0.0;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        bool var_107;
        bool var_108;
        wp::int32* var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::int32 var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::int32* var_114;
        wp::vec_t<3, wp::float32>* var_115;
        wp::int32 var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::float32 var_119;
        wp::int32* var_120;
        wp::vec_t<3, wp::float32>* var_121;
        wp::int32 var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::int32* var_125;
        wp::vec_t<3, wp::float32>* var_126;
        wp::int32 var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::float32 var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::float32 var_133;
        wp::array_t<wp::float32> var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        const wp::float32 var_137 = 1e-15;
        bool var_138;
        const wp::float32 var_139 = 1.0;
        const wp::float32 var_140 = 0.0;
        const wp::float32 var_141 = 0.0;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        bool var_144;
        wp::int32* var_145;
        wp::vec_t<3, wp::float32>* var_146;
        wp::int32 var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::int32* var_150;
        wp::vec_t<3, wp::float32>* var_151;
        wp::int32 var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        //---------
        // forward
        // def _spatial_geom_tendon(                                                              <L 3823>
        // worldid, elementid = wp.tid()                                                          <L 3853>
        builtin_tid2d(var_0, var_1);
        // wrap_adr = wrap_geom_adr[elementid]                                                    <L 3854>
        var_2 = wp::address(var_wrap_geom_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tenid = tendon_geom_adr[elementid]                                                     <L 3855>
        var_5 = wp::address(var_tendon_geom_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // pulley_scale = wrap_pulley_scale[wrap_adr]                                             <L 3858>
        var_8 = wp::address(var_wrap_pulley_scale, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // wrap_objid_site0 = wrap_objid[wrap_adr - 1]                                            <L 3861>
        var_12 = wp::sub(var_3, var_11);
        var_13 = wp::address(var_wrap_objid, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // wrap_objid_geom = wrap_objid[wrap_adr + 0]                                             <L 3862>
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::address(var_wrap_objid, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // wrap_objid_site1 = wrap_objid[wrap_adr + 1]                                            <L 3863>
        var_22 = wp::add(var_3, var_21);
        var_23 = wp::address(var_wrap_objid, var_22);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // site_pnt0 = site_xpos_in[worldid, wrap_objid_site0]                                    <L 3866>
        var_26 = wp::address(var_site_xpos_in, var_0, var_14);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // site_pnt1 = site_xpos_in[worldid, wrap_objid_site1]                                    <L 3867>
        var_29 = wp::address(var_site_xpos_in, var_0, var_24);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // geom_xpos = geom_xpos_in[worldid, wrap_objid_geom]                                     <L 3870>
        var_32 = wp::address(var_geom_xpos_in, var_0, var_19);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // geom_xmat = geom_xmat_in[worldid, wrap_objid_geom]                                     <L 3871>
        var_35 = wp::address(var_geom_xmat_in, var_0, var_19);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // geomsize = geom_size[worldid % geom_size.shape[0], wrap_objid_geom][0]                 <L 3872>
        var_38 = &(var_geom_size.shape);
        var_41 = wp::load(var_38);
        var_40 = wp::extract(var_41, var_39);
        var_42 = wp::mod(var_0, var_40);
        var_43 = wp::address(var_geom_size, var_42, var_19);
        var_46 = wp::load(var_43);
        var_45 = wp::extract(var_46, var_44);
        // geom_type = wrap_type[wrap_adr]                                                        <L 3873>
        var_47 = wp::address(var_wrap_type, var_3);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // bodyid_site0 = site_bodyid[wrap_objid_site0]                                           <L 3876>
        var_50 = wp::address(var_site_bodyid, var_14);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // bodyid_geom = geom_bodyid[wrap_objid_geom]                                             <L 3877>
        var_53 = wp::address(var_geom_bodyid, var_19);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // bodyid_site1 = site_bodyid[wrap_objid_site1]                                           <L 3878>
        var_56 = wp::address(var_site_bodyid, var_24);
        var_58 = wp::load(var_56);
        var_57 = wp::copy(var_58);
        // sideid = int(wp.round(wrap_prm[wrap_adr]))                                             <L 3881>
        var_59 = wp::address(var_wrap_prm, var_3);
        var_61 = wp::load(var_59);
        var_60 = wp::round(var_61);
        var_62 = wp::int(var_60);
        // if sideid >= 0:                                                                        <L 3882>
        var_64 = (var_62 >= var_63);
        if (var_64) {
            // side = site_xpos_in[worldid, sideid]                                               <L 3883>
            var_65 = wp::address(var_site_xpos_in, var_0, var_62);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
        }
        if (!var_64) {
            // side = wp.vec3(MJ_MAXVAL)                                                          <L 3885>
            var_69 = wp::vec_t<3, wp::float32>(var_68);
        }
        var_70 = wp::where(var_64, var_66, var_69);
        // length_geomgeom, geom_pnt0, geom_pnt1 = util_misc.wrap(site_pnt0, site_pnt1, geom_xpos, geom_xmat, geomsize, geom_type, side)       <L 3888>
        wrap_0(var_27, var_30, var_33, var_36, var_45, var_48, var_70, var_71, var_72, var_73);
        // wrap_geom_xpos_out[worldid, elementid] = wp.spatial_vector(geom_pnt0, geom_pnt1)       <L 3891>
        var_74 = wp::vec_t<6, wp::float32>(var_72, var_73);
        wp::array_store(var_wrap_geom_xpos_out, var_0, var_1, var_74);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 3893>
        var_75 = wp::address(var_ten_J_rownnz, var_6);
        var_77 = wp::load(var_75);
        var_76 = wp::copy(var_77);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 3894>
        var_78 = wp::address(var_ten_J_rowadr, var_6);
        var_80 = wp::load(var_78);
        var_79 = wp::copy(var_80);
        // if length_geomgeom >= 0.0:                                                             <L 3896>
        var_82 = (var_71 >= var_81);
        if (var_82) {
            // dif_sitegeom = geom_pnt0 - site_pnt0                                               <L 3897>
            var_83 = wp::sub(var_72, var_27);
            // dif_geomsite = site_pnt1 - geom_pnt1                                               <L 3898>
            var_84 = wp::sub(var_30, var_73);
            // vec_sitegeom, length_sitegeom = math.normalize_with_norm(dif_sitegeom)             <L 3899>
            normalize_with_norm_0(var_83, var_85, var_86);
            // vec_geomsite, length_geomsite = math.normalize_with_norm(dif_geomsite)             <L 3900>
            normalize_with_norm_0(var_84, var_87, var_88);
            // length_sitegeomsite = length_sitegeom + length_geomgeom + length_geomsite          <L 3903>
            var_89 = wp::add(var_86, var_71);
            var_90 = wp::add(var_89, var_88);
            // if length_sitegeomsite:                                                            <L 3905>
            if (var_90) {
                // wp.atomic_add(ten_length_out[worldid], tenid, length_sitegeomsite * pulley_scale)       <L 3906>
                var_91 = wp::view(var_ten_length_out, var_0);
                var_92 = wp::mul(var_90, var_9);
                var_93 = wp::atomic_add(var_91, var_6, var_92);
            }
            // if length_sitegeom < MJ_MINVAL:                                                    <L 3909>
            var_95 = (var_86 < var_94);
            if (var_95) {
                // vec_sitegeom = wp.vec3(1.0, 0.0, 0.0)                                          <L 3910>
                var_99 = wp::vec_t<3, wp::float32>(var_96, var_97, var_98);
            }
            var_100 = wp::where(var_95, var_99, var_85);
            // if length_geomsite < MJ_MINVAL:                                                    <L 3912>
            var_101 = (var_88 < var_94);
            if (var_101) {
                // vec_geomsite = wp.vec3(1.0, 0.0, 0.0)                                          <L 3913>
                var_105 = wp::vec_t<3, wp::float32>(var_102, var_103, var_104);
            }
            var_106 = wp::where(var_101, var_105, var_87);
            // dif_body_sitegeom = bodyid_site0 != bodyid_geom                                    <L 3915>
            var_107 = (var_51 != var_54);
            // dif_body_geomsite = bodyid_geom != bodyid_site1                                    <L 3916>
            var_108 = (var_54 != var_57);
            // if dif_body_sitegeom:                                                              <L 3919>
            if (var_107) {
                // offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]       <L 3920>
                var_109 = wp::address(var_body_rootid, var_51);
                var_111 = wp::load(var_109);
                var_110 = wp::address(var_subtree_com_in, var_0, var_111);
                var_113 = wp::load(var_110);
                var_112 = wp::sub(var_27, var_113);
                // offset_geom0 = geom_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_geom]]       <L 3921>
                var_114 = wp::address(var_body_rootid, var_54);
                var_116 = wp::load(var_114);
                var_115 = wp::address(var_subtree_com_in, var_0, var_116);
                var_118 = wp::load(var_115);
                var_117 = wp::sub(var_72, var_118);
                // _accumulate_jac_chain(                                                         <L 3922>
                // body_parentid,                                                                 <L 3923>
                // body_dofnum,                                                                   <L 3924>
                // body_dofadr,                                                                   <L 3925>
                // ten_J_colind,                                                                  <L 3926>
                // cdof_in,                                                                       <L 3927>
                // offset_site0,                                                                  <L 3928>
                // vec_sitegeom,                                                                  <L 3929>
                // bodyid_site0,                                                                  <L 3930>
                // rowadr,                                                                        <L 3931>
                // rownnz,                                                                        <L 3932>
                // -pulley_scale,                                                                 <L 3933>
                var_119 = wp::neg(var_9);
                // worldid,                                                                       <L 3934>
                // ten_J_out,                                                                     <L 3935>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_112, var_100, var_51, var_79, var_76, var_119, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 3937>
                // body_parentid,                                                                 <L 3938>
                // body_dofnum,                                                                   <L 3939>
                // body_dofadr,                                                                   <L 3940>
                // ten_J_colind,                                                                  <L 3941>
                // cdof_in,                                                                       <L 3942>
                // offset_geom0,                                                                  <L 3943>
                // vec_sitegeom,                                                                  <L 3944>
                // bodyid_geom,                                                                   <L 3945>
                // rowadr,                                                                        <L 3946>
                // rownnz,                                                                        <L 3947>
                // pulley_scale,                                                                  <L 3948>
                // worldid,                                                                       <L 3949>
                // ten_J_out,                                                                     <L 3950>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_117, var_100, var_54, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
            // if dif_body_geomsite:                                                              <L 3954>
            if (var_108) {
                // offset_geom1 = geom_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_geom]]       <L 3955>
                var_120 = wp::address(var_body_rootid, var_54);
                var_122 = wp::load(var_120);
                var_121 = wp::address(var_subtree_com_in, var_0, var_122);
                var_124 = wp::load(var_121);
                var_123 = wp::sub(var_73, var_124);
                // offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]       <L 3956>
                var_125 = wp::address(var_body_rootid, var_57);
                var_127 = wp::load(var_125);
                var_126 = wp::address(var_subtree_com_in, var_0, var_127);
                var_129 = wp::load(var_126);
                var_128 = wp::sub(var_30, var_129);
                // _accumulate_jac_chain(                                                         <L 3957>
                // body_parentid,                                                                 <L 3958>
                // body_dofnum,                                                                   <L 3959>
                // body_dofadr,                                                                   <L 3960>
                // ten_J_colind,                                                                  <L 3961>
                // cdof_in,                                                                       <L 3962>
                // offset_geom1,                                                                  <L 3963>
                // vec_geomsite,                                                                  <L 3964>
                // bodyid_geom,                                                                   <L 3965>
                // rowadr,                                                                        <L 3966>
                // rownnz,                                                                        <L 3967>
                // -pulley_scale,                                                                 <L 3968>
                var_130 = wp::neg(var_9);
                // worldid,                                                                       <L 3969>
                // ten_J_out,                                                                     <L 3970>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_123, var_106, var_54, var_79, var_76, var_130, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 3972>
                // body_parentid,                                                                 <L 3973>
                // body_dofnum,                                                                   <L 3974>
                // body_dofadr,                                                                   <L 3975>
                // ten_J_colind,                                                                  <L 3976>
                // cdof_in,                                                                       <L 3977>
                // offset_site1,                                                                  <L 3978>
                // vec_geomsite,                                                                  <L 3979>
                // bodyid_site1,                                                                  <L 3980>
                // rowadr,                                                                        <L 3981>
                // rownnz,                                                                        <L 3982>
                // pulley_scale,                                                                  <L 3983>
                // worldid,                                                                       <L 3984>
                // ten_J_out,                                                                     <L 3985>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_128, var_106, var_57, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
        }
        if (!var_82) {
            // dif_sitesite = site_pnt1 - site_pnt0                                               <L 3988>
            var_131 = wp::sub(var_30, var_27);
            // vec_sitesite, length_sitesite = math.normalize_with_norm(dif_sitesite)             <L 3989>
            normalize_with_norm_0(var_131, var_132, var_133);
            // if length_sitesite:                                                                <L 3992>
            if (var_133) {
                // wp.atomic_add(ten_length_out[worldid], tenid, length_sitesite * pulley_scale)       <L 3993>
                var_134 = wp::view(var_ten_length_out, var_0);
                var_135 = wp::mul(var_133, var_9);
                var_136 = wp::atomic_add(var_134, var_6, var_135);
            }
            // if length_sitesite < MJ_MINVAL:                                                    <L 3996>
            var_138 = (var_133 < var_137);
            if (var_138) {
                // vec_sitesite = wp.vec3(1.0, 0.0, 0.0)                                          <L 3997>
                var_142 = wp::vec_t<3, wp::float32>(var_139, var_140, var_141);
            }
            var_143 = wp::where(var_138, var_142, var_132);
            // if bodyid_site0 != bodyid_site1:                                                   <L 3999>
            var_144 = (var_51 != var_57);
            if (var_144) {
                // offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]       <L 4000>
                var_145 = wp::address(var_body_rootid, var_51);
                var_147 = wp::load(var_145);
                var_146 = wp::address(var_subtree_com_in, var_0, var_147);
                var_149 = wp::load(var_146);
                var_148 = wp::sub(var_27, var_149);
                // offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]       <L 4001>
                var_150 = wp::address(var_body_rootid, var_57);
                var_152 = wp::load(var_150);
                var_151 = wp::address(var_subtree_com_in, var_0, var_152);
                var_154 = wp::load(var_151);
                var_153 = wp::sub(var_30, var_154);
                // _accumulate_jac_chain(                                                         <L 4002>
                // body_parentid,                                                                 <L 4003>
                // body_dofnum,                                                                   <L 4004>
                // body_dofadr,                                                                   <L 4005>
                // ten_J_colind,                                                                  <L 4006>
                // cdof_in,                                                                       <L 4007>
                // offset_site0,                                                                  <L 4008>
                // vec_sitesite,                                                                  <L 4009>
                // bodyid_site0,                                                                  <L 4010>
                // rowadr,                                                                        <L 4011>
                // rownnz,                                                                        <L 4012>
                // -pulley_scale,                                                                 <L 4013>
                var_155 = wp::neg(var_9);
                // worldid,                                                                       <L 4014>
                // ten_J_out,                                                                     <L 4015>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_148, var_143, var_51, var_79, var_76, var_155, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 4017>
                // body_parentid,                                                                 <L 4018>
                // body_dofnum,                                                                   <L 4019>
                // body_dofadr,                                                                   <L 4020>
                // ten_J_colind,                                                                  <L 4021>
                // cdof_in,                                                                       <L 4022>
                // offset_site1,                                                                  <L 4023>
                // vec_sitesite,                                                                  <L 4024>
                // bodyid_site1,                                                                  <L 4025>
                // rowadr,                                                                        <L 4026>
                // rownnz,                                                                        <L 4027>
                // pulley_scale,                                                                  <L 4028>
                // worldid,                                                                       <L 4029>
                // ten_J_out,                                                                     <L 4030>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_153, var_143, var_57, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
        }
        var_156 = wp::where(var_82, var_94, var_137);
        var_157 = wp::where(var_82, var_112, var_148);
        var_158 = wp::where(var_82, var_128, var_153);
    }
}



extern "C" __global__ void _subtree_vel_forward_1d7ed456_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_subtree_bodyvel_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::mat_t<3, 3, wp::float32>* var_20;
        wp::mat_t<3, 3, wp::float32> var_21;
        wp::mat_t<3, 3, wp::float32> var_22;
        wp::int32* var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::float32* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::float32 var_33;
        wp::mat_t<3, 3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        const wp::int32 var_37 = 0;
        wp::float32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        const wp::int32 var_40 = 0;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32>* var_43;
        const wp::int32 var_44 = 1;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 1;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::vec_t<3, wp::float32>* var_50;
        const wp::int32 var_51 = 2;
        wp::float32 var_52;
        wp::vec_t<3, wp::float32> var_53;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<6, wp::float32> var_58;
        //---------
        // forward
        // def _subtree_vel_forward(                                                              <L 3499>
        // worldid, bodyid = wp.tid()                                                             <L 3515>
        builtin_tid2d(var_0, var_1);
        // body_mass_id = worldid % body_mass.shape[0]                                            <L 3516>
        var_2 = &(var_body_mass.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_inertia_id = worldid % body_inertia.shape[0]                                      <L 3517>
        var_7 = &(var_body_inertia.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 3519>
        var_12 = wp::address(var_cvel_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // ang = wp.spatial_top(cvel)                                                             <L 3520>
        var_15 = wp::spatial_top(var_13);
        // lin = wp.spatial_bottom(cvel)                                                          <L 3521>
        var_16 = wp::spatial_bottom(var_13);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 3522>
        var_17 = wp::address(var_xipos_in, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // ximat = ximat_in[worldid, bodyid]                                                      <L 3523>
        var_20 = wp::address(var_ximat_in, var_0, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // subtree_com_root = subtree_com_in[worldid, body_rootid[bodyid]]                        <L 3524>
        var_23 = wp::address(var_body_rootid, var_1);
        var_25 = wp::load(var_23);
        var_24 = wp::address(var_subtree_com_in, var_0, var_25);
        var_27 = wp::load(var_24);
        var_26 = wp::copy(var_27);
        // lin -= wp.cross(xipos - subtree_com_root, ang)                                         <L 3527>
        var_28 = wp::sub(var_18, var_26);
        var_29 = wp::cross(var_28, var_15);
        var_30 = wp::sub(var_16, var_29);
        // subtree_linvel_out[worldid, bodyid] = body_mass[body_mass_id, bodyid] * lin            <L 3529>
        var_31 = wp::address(var_body_mass, var_6, var_1);
        var_33 = wp::load(var_31);
        var_32 = wp::mul(var_33, var_30);
        wp::array_store(var_subtree_linvel_out, var_0, var_1, var_32);
        // dv = wp.transpose(ximat) @ ang                                                         <L 3530>
        var_34 = wp::transpose(var_21);
        var_35 = wp::mul(var_34, var_15);
        // dv[0] *= body_inertia[body_inertia_id, bodyid][0]                                      <L 3531>
        var_36 = wp::address(var_body_inertia, var_11, var_1);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_41 = wp::extract(var_35, var_40);
        var_42 = wp::mul(var_41, var_38);
        wp::assign_inplace(var_35, var_40, var_42);
        // dv[1] *= body_inertia[body_inertia_id, bodyid][1]                                      <L 3532>
        var_43 = wp::address(var_body_inertia, var_11, var_1);
        var_46 = wp::load(var_43);
        var_45 = wp::extract(var_46, var_44);
        var_48 = wp::extract(var_35, var_47);
        var_49 = wp::mul(var_48, var_45);
        wp::assign_inplace(var_35, var_47, var_49);
        // dv[2] *= body_inertia[body_inertia_id, bodyid][2]                                      <L 3533>
        var_50 = wp::address(var_body_inertia, var_11, var_1);
        var_53 = wp::load(var_50);
        var_52 = wp::extract(var_53, var_51);
        var_55 = wp::extract(var_35, var_54);
        var_56 = wp::mul(var_55, var_52);
        wp::assign_inplace(var_35, var_54, var_56);
        // subtree_angmom_out[worldid, bodyid] = ximat @ dv                                       <L 3534>
        var_57 = wp::mul(var_21, var_35);
        wp::array_store(var_subtree_angmom_out, var_0, var_1, var_57);
        // subtree_bodyvel_out[worldid, bodyid] = wp.spatial_vector(ang, lin)                     <L 3535>
        var_58 = wp::vec_t<6, wp::float32>(var_15, var_30);
        wp::array_store(var_subtree_bodyvel_out, var_0, var_1, var_58);
    }
}



extern "C" __global__ void _tendon_bias_coef_2556dfce_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_Jdot_in,
    wp::array_t<wp::float32> var_ten_bias_coef_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::array_t<wp::float32> var_29;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        //---------
        // forward
        // def _tendon_bias_coef(                                                                 <L 2051>
        // worldid, tenid, dofid_sparse = wp.tid()                                                <L 2064>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 2066>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 2067>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 2068>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 2070>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid_sparse >= rownnz:                                                             <L 2071>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 2072>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 2073>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // sparseid = rowadr + dofid_sparse                                                       <L 2074>
        var_20 = wp::add(var_18, var_2);
        // ten_Jdot = ten_Jdot_in[worldid, sparseid]                                              <L 2075>
        var_21 = wp::address(var_ten_Jdot_in, var_0, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if ten_Jdot == 0.0:                                                                    <L 2076>
        var_25 = (var_22 == var_24);
        if (var_25) {
            // return                                                                             <L 2077>
            continue;
        }
        // dofid = ten_J_colind[sparseid]                                                         <L 2079>
        var_26 = wp::address(var_ten_J_colind, var_20);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_add(ten_bias_coef_out[worldid], tenid, ten_Jdot * qvel_in[worldid, dofid])       <L 2080>
        var_29 = wp::view(var_ten_bias_coef_out, var_0);
        var_30 = wp::address(var_qvel_in, var_0, var_27);
        var_32 = wp::load(var_30);
        var_31 = wp::mul(var_22, var_32);
        var_33 = wp::atomic_add(var_29, var_1, var_31);
    }
}



extern "C" __global__ void _compute_body_matrices_27760750_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::quat_t<wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::quat_t<wp::float32> var_4;
        //---------
        // forward
        // def _compute_body_matrices(                                                            <L 168>
        // worldid, bodyid = wp.tid()                                                             <L 174>
        builtin_tid2d(var_0, var_1);
        // xmat_out[worldid, bodyid] = math.quat_to_mat(xquat_in[worldid, bodyid])                <L 175>
        var_2 = wp::address(var_xquat_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = quat_to_mat_0(var_4);
        wp::array_store(var_xmat_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _tendon_bias_qfrc_2ae21571_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_bias_coef_in,
    wp::array_t<wp::float32> var_qfrc_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::array_t<wp::float32> var_29;
        wp::float32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        //---------
        // forward
        // def _tendon_bias_qfrc(                                                                 <L 2084>
        // worldid, tenid, dofid = wp.tid()                                                       <L 2097>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 2099>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 2100>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 2101>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 2103>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid >= rownnz:                                                                    <L 2104>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 2105>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 2106>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // sparseid = rowadr + dofid                                                              <L 2107>
        var_20 = wp::add(var_18, var_2);
        // ten_J = ten_J_in[worldid, sparseid]                                                    <L 2108>
        var_21 = wp::address(var_ten_J_in, var_0, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if ten_J == 0.0:                                                                       <L 2110>
        var_25 = (var_22 == var_24);
        if (var_25) {
            // return                                                                             <L 2111>
            continue;
        }
        // dofid = ten_J_colind[sparseid]                                                         <L 2113>
        var_26 = wp::address(var_ten_J_colind, var_20);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_add(qfrc_out[worldid], dofid, ten_J * armature * ten_bias_coef_in[worldid, tenid])       <L 2115>
        var_29 = wp::view(var_qfrc_out, var_0);
        var_30 = wp::mul(var_22, var_9);
        var_31 = wp::address(var_ten_bias_coef_in, var_0, var_1);
        var_33 = wp::load(var_31);
        var_32 = wp::mul(var_30, var_33);
        var_34 = wp::atomic_add(var_29, var_27, var_32);
    }
}



extern "C" __global__ void _spatial_tendon_wrap_2bb6c667_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_ntendon,
    wp::array_t<wp::int32> var_tendon_adr,
    wp::array_t<wp::int32> var_tendon_num,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_geom_xpos_in,
    wp::array_t<wp::int32> var_ten_wrapadr_out,
    wp::array_t<wp::int32> var_ten_wrapnum_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_wrap_obj_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_xpos_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32 var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        bool var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 2;
        bool var_49;
        bool var_50;
        const wp::int32 var_51 = 2;
        bool var_52;
        const wp::int32 var_53 = 2;
        wp::int32 var_54;
        const wp::int32 var_55 = 2;
        wp::int32 var_56;
        const wp::float32 var_57 = 0.0;
        const wp::int32 var_58 = 3;
        wp::int32 var_59;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        const wp::float32 var_62 = 0.0;
        const wp::int32 var_63 = 3;
        wp::int32 var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        const wp::float32 var_67 = 0.0;
        const wp::int32 var_68 = 3;
        wp::int32 var_69;
        const wp::int32 var_70 = 2;
        wp::int32 var_71;
        const wp::int32 var_72 = -2;
        const wp::int32 var_73 = 1;
        wp::int32 var_74;
        const wp::int32 var_75 = 1;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        wp::vec_t<3, wp::float32>* var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        bool var_84;
        const wp::int32 var_85 = 4;
        bool var_86;
        const wp::int32 var_87 = 5;
        bool var_88;
        wp::vec_t<6, wp::float32>* var_89;
        wp::vec_t<6, wp::float32> var_90;
        wp::vec_t<6, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        wp::float32 var_102;
        const wp::float32 var_103 = 10000000000.0;
        bool var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32>* var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        const wp::int32 var_109 = 0;
        wp::int32 var_110;
        const wp::int32 var_111 = 2;
        wp::int32 var_112;
        const wp::int32 var_113 = 0;
        wp::int32 var_114;
        const wp::int32 var_115 = 2;
        wp::int32 var_116;
        const wp::int32 var_117 = 1;
        wp::int32 var_118;
        const wp::int32 var_119 = 2;
        wp::int32 var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        const wp::int32 var_123 = 2;
        wp::int32 var_124;
        const wp::int32 var_125 = 2;
        wp::int32 var_126;
        const wp::int32 var_127 = 2;
        wp::int32 var_128;
        const wp::int32 var_129 = 2;
        wp::int32 var_130;
        const wp::int32 var_131 = 2;
        wp::int32 var_132;
        const wp::int32 var_133 = 3;
        wp::int32 var_134;
        const wp::int32 var_135 = 2;
        wp::int32 var_136;
        const wp::int32 var_137 = 3;
        wp::int32 var_138;
        const wp::int32 var_139 = 2;
        wp::int32 var_140;
        const wp::int32 var_141 = 0;
        wp::float32 var_142;
        const wp::int32 var_143 = 3;
        wp::int32 var_144;
        const wp::int32 var_145 = 0;
        wp::int32 var_146;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        const wp::int32 var_149 = 3;
        wp::int32 var_150;
        const wp::int32 var_151 = 1;
        wp::int32 var_152;
        const wp::int32 var_153 = 2;
        wp::float32 var_154;
        const wp::int32 var_155 = 3;
        wp::int32 var_156;
        const wp::int32 var_157 = 2;
        wp::int32 var_158;
        const wp::int32 var_159 = 0;
        wp::float32 var_160;
        const wp::int32 var_161 = 3;
        wp::int32 var_162;
        const wp::int32 var_163 = 0;
        wp::int32 var_164;
        const wp::int32 var_165 = 1;
        wp::float32 var_166;
        const wp::int32 var_167 = 3;
        wp::int32 var_168;
        const wp::int32 var_169 = 1;
        wp::int32 var_170;
        const wp::int32 var_171 = 2;
        wp::float32 var_172;
        const wp::int32 var_173 = 3;
        wp::int32 var_174;
        const wp::int32 var_175 = 2;
        wp::int32 var_176;
        const wp::int32 var_177 = 0;
        wp::float32 var_178;
        const wp::int32 var_179 = 3;
        wp::int32 var_180;
        const wp::int32 var_181 = 0;
        wp::int32 var_182;
        const wp::int32 var_183 = 1;
        wp::float32 var_184;
        const wp::int32 var_185 = 3;
        wp::int32 var_186;
        const wp::int32 var_187 = 1;
        wp::int32 var_188;
        const wp::int32 var_189 = 2;
        wp::float32 var_190;
        const wp::int32 var_191 = 3;
        wp::int32 var_192;
        const wp::int32 var_193 = 2;
        wp::int32 var_194;
        const wp::int32 var_195 = 0;
        wp::float32 var_196;
        const wp::int32 var_197 = 3;
        wp::int32 var_198;
        const wp::int32 var_199 = 0;
        wp::int32 var_200;
        const wp::int32 var_201 = 1;
        wp::float32 var_202;
        const wp::int32 var_203 = 3;
        wp::int32 var_204;
        const wp::int32 var_205 = 1;
        wp::int32 var_206;
        const wp::int32 var_207 = 2;
        wp::float32 var_208;
        const wp::int32 var_209 = 3;
        wp::int32 var_210;
        const wp::int32 var_211 = 2;
        wp::int32 var_212;
        const wp::int32 var_213 = -1;
        const wp::int32 var_214 = 3;
        wp::int32 var_215;
        const wp::int32 var_216 = 3;
        wp::int32 var_217;
        const wp::int32 var_218 = 2;
        wp::int32 var_219;
        const wp::int32 var_220 = 0;
        wp::int32 var_221;
        const wp::int32 var_222 = 2;
        wp::int32 var_223;
        const wp::int32 var_224 = 0;
        wp::int32 var_225;
        const wp::int32 var_226 = 2;
        wp::int32 var_227;
        const wp::int32 var_228 = 0;
        wp::float32 var_229;
        const wp::int32 var_230 = 3;
        wp::int32 var_231;
        const wp::int32 var_232 = 0;
        wp::int32 var_233;
        const wp::int32 var_234 = 1;
        wp::float32 var_235;
        const wp::int32 var_236 = 3;
        wp::int32 var_237;
        const wp::int32 var_238 = 1;
        wp::int32 var_239;
        const wp::int32 var_240 = 2;
        wp::float32 var_241;
        const wp::int32 var_242 = 3;
        wp::int32 var_243;
        const wp::int32 var_244 = 2;
        wp::int32 var_245;
        const wp::int32 var_246 = -1;
        const wp::int32 var_247 = 1;
        wp::int32 var_248;
        const wp::int32 var_249 = 1;
        wp::int32 var_250;
        const wp::int32 var_251 = 2;
        wp::int32 var_252;
        wp::int32 var_253;
        wp::int32 var_254;
        wp::int32 var_255;
        wp::int32 var_256;
        wp::int32 var_257;
        const wp::int32 var_258 = 0;
        wp::int32 var_259;
        const wp::int32 var_260 = 2;
        wp::int32 var_261;
        const wp::int32 var_262 = 0;
        wp::int32 var_263;
        const wp::int32 var_264 = 2;
        wp::int32 var_265;
        const wp::int32 var_266 = 0;
        wp::float32 var_267;
        const wp::int32 var_268 = 3;
        wp::int32 var_269;
        const wp::int32 var_270 = 0;
        wp::int32 var_271;
        const wp::int32 var_272 = 1;
        wp::float32 var_273;
        const wp::int32 var_274 = 3;
        wp::int32 var_275;
        const wp::int32 var_276 = 1;
        wp::int32 var_277;
        const wp::int32 var_278 = 2;
        wp::float32 var_279;
        const wp::int32 var_280 = 3;
        wp::int32 var_281;
        const wp::int32 var_282 = 2;
        wp::int32 var_283;
        const wp::int32 var_284 = -1;
        const wp::int32 var_285 = 1;
        wp::int32 var_286;
        const wp::int32 var_287 = 1;
        wp::int32 var_288;
        const wp::int32 var_289 = 1;
        wp::int32 var_290;
        wp::int32 var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        wp::int32 var_296;
        wp::int32 var_297;
        wp::int32 var_298;
        const wp::int32 var_299 = 1;
        wp::int32 var_300;
        wp::shape_t* var_301;
        const wp::int32 var_302 = 0;
        wp::int32 var_303;
        wp::shape_t var_304;
        bool var_305;
        wp::int32 var_306;
        const wp::int32 var_307 = 1;
        wp::int32 var_308;
        wp::int32* var_309;
        const wp::int32 var_310 = 2;
        bool var_311;
        wp::int32 var_312;
        const bool var_313 = false;
        bool var_314;
        bool var_315;
        const wp::int32 var_316 = 1;
        wp::int32 var_317;
        bool var_318;
        const wp::int32 var_319 = 0;
        wp::int32 var_320;
        const wp::int32 var_321 = 2;
        wp::int32 var_322;
        const wp::int32 var_323 = 0;
        wp::int32 var_324;
        const wp::int32 var_325 = 2;
        wp::int32 var_326;
        wp::vec_t<3, wp::float32>* var_327;
        wp::vec_t<3, wp::float32> var_328;
        wp::vec_t<3, wp::float32> var_329;
        const wp::int32 var_330 = 0;
        wp::float32 var_331;
        const wp::int32 var_332 = 3;
        wp::int32 var_333;
        const wp::int32 var_334 = 0;
        wp::int32 var_335;
        const wp::int32 var_336 = 1;
        wp::float32 var_337;
        const wp::int32 var_338 = 3;
        wp::int32 var_339;
        const wp::int32 var_340 = 1;
        wp::int32 var_341;
        const wp::int32 var_342 = 2;
        wp::float32 var_343;
        const wp::int32 var_344 = 3;
        wp::int32 var_345;
        const wp::int32 var_346 = 2;
        wp::int32 var_347;
        const wp::int32 var_348 = -1;
        const wp::int32 var_349 = 1;
        wp::int32 var_350;
        const wp::int32 var_351 = 1;
        wp::int32 var_352;
        wp::int32 var_353;
        wp::int32 var_354;
        wp::vec_t<3, wp::float32> var_355;
        wp::int32 var_356;
        wp::int32 var_357;
        //---------
        // forward
        // def _spatial_tendon_wrap(                                                              <L 4035>
        // worldid = wp.tid()                                                                     <L 4052>
        var_0 = builtin_tid1d();
        // wrapcount = int(0)                                                                     <L 4054>
        var_2 = wp::int(var_1);
        // wrapgeomid = int(0)                                                                    <L 4055>
        var_4 = wp::int(var_3);
        // for i in range(ntendon):                                                               <L 4056>
        var_5 = wp::range(var_ntendon);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // adr = tendon_adr[i]                                                                <L 4057>
            var_7 = wp::address(var_tendon_adr, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // ten_wrapadr_out[worldid, i] = wrapcount                                            <L 4058>
            wp::array_store(var_ten_wrapadr_out, var_0, var_6, var_2);
            // wrapnum = int(0)                                                                   <L 4059>
            var_11 = wp::int(var_10);
            // tendonnum = tendon_num[i]                                                          <L 4060>
            var_12 = wp::address(var_tendon_num, var_6);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // if wrap_type[adr] == WrapType.JOINT:                                               <L 4063>
            var_15 = wp::address(var_wrap_type, var_8);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            if (var_17) {
                // continue                                                                       <L 4064>
                goto start_for_0;
            }
            // j = int(0)                                                                         <L 4067>
            var_20 = wp::int(var_19);
            // while j < tendonnum - 1:                                                           <L 4068>
        start_while_2:;
            var_22 = wp::sub(var_13, var_21);
            var_23 = (var_20 < var_22);
        if ((var_23) == false) goto end_while_2;
                // type0 = wrap_type[adr + j + 0]                                                 <L 4070>
                var_24 = wp::add(var_8, var_20);
                var_26 = wp::add(var_24, var_25);
                var_27 = wp::address(var_wrap_type, var_26);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // type1 = wrap_type[adr + j + 1]                                                 <L 4071>
                var_30 = wp::add(var_8, var_20);
                var_32 = wp::add(var_30, var_31);
                var_33 = wp::address(var_wrap_type, var_32);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // id0 = wrap_objid[adr + j + 0]                                                  <L 4072>
                var_36 = wp::add(var_8, var_20);
                var_38 = wp::add(var_36, var_37);
                var_39 = wp::address(var_wrap_objid, var_38);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // id1 = wrap_objid[adr + j + 1]                                                  <L 4073>
                var_42 = wp::add(var_8, var_20);
                var_44 = wp::add(var_42, var_43);
                var_45 = wp::address(var_wrap_objid, var_44);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // pulley0 = type0 == WrapType.PULLEY                                             <L 4076>
                var_49 = (var_28 == var_48);
                // if pulley0 or type1 == WrapType.PULLEY:                                        <L 4077>
                var_50 = var_49;
                if (!var_50) {
                    var_52 = (var_34 == var_51);
                    var_50 = var_50 || var_52;
                }
                if (var_50) {
                    // if pulley0:                                                                <L 4078>
                    if (var_49) {
                        // row = wrapcount // 2                                                   <L 4079>
                        var_54 = wp::floordiv(var_2, var_53);
                        // col = wrapcount % 2                                                    <L 4080>
                        var_56 = wp::mod(var_2, var_55);
                        // wrap_xpos_out[worldid, row][3 * col + 0] = 0.0                         <L 4081>
                        var_59 = wp::mul(var_58, var_56);
                        var_61 = wp::add(var_59, var_60);
                        wp::index(var_wrap_xpos_out, var_0, var_54)[var_61] = var_57;
                        // wrap_xpos_out[worldid, row][3 * col + 1] = 0.0                         <L 4082>
                        var_64 = wp::mul(var_63, var_56);
                        var_66 = wp::add(var_64, var_65);
                        wp::index(var_wrap_xpos_out, var_0, var_54)[var_66] = var_62;
                        // wrap_xpos_out[worldid, row][3 * col + 2] = 0.0                         <L 4083>
                        var_69 = wp::mul(var_68, var_56);
                        var_71 = wp::add(var_69, var_70);
                        wp::index(var_wrap_xpos_out, var_0, var_54)[var_71] = var_67;
                        // wrap_obj_out[worldid, row][col] = -2                                   <L 4085>
                        wp::index(var_wrap_obj_out, var_0, var_54)[var_56] = var_72;
                        // wrapnum += 1                                                           <L 4087>
                        var_74 = wp::add(var_11, var_73);
                        // wrapcount += 1                                                         <L 4088>
                        var_76 = wp::add(var_2, var_75);
                    }
                    var_77 = wp::where(var_49, var_76, var_2);
                    var_78 = wp::where(var_49, var_74, var_11);
                    // j += 1                                                                     <L 4091>
                    var_80 = wp::add(var_20, var_79);
                    // continue                                                                   <L 4092>
                    wp::assign(var_2, var_77);
                    wp::assign(var_11, var_78);
                    wp::assign(var_20, var_80);
                    goto start_while_2;
                }
                // wpnt_site0 = site_xpos_in[worldid, id0]                                        <L 4095>
                var_81 = wp::address(var_site_xpos_in, var_0, var_40);
                var_83 = wp::load(var_81);
                var_82 = wp::copy(var_83);
                // if type1 == WrapType.SPHERE or type1 == WrapType.CYLINDER:                     <L 4098>
                var_86 = (var_34 == var_85);
                var_84 = var_86;
                if (!var_84) {
                    var_88 = (var_34 == var_87);
                    var_84 = var_84 || var_88;
                }
                if (var_84) {
                    // wrap_geom_xpos = wrap_geom_xpos_in[worldid, wrapgeomid]                    <L 4099>
                    var_89 = wp::address(var_wrap_geom_xpos_in, var_0, var_4);
                    var_91 = wp::load(var_89);
                    var_90 = wp::copy(var_91);
                    // wpnt_geom0 = wp.spatial_top(wrap_geom_xpos)                                <L 4100>
                    var_92 = wp::spatial_top(var_90);
                    // wrapgeomid += 1                                                            <L 4101>
                    var_94 = wp::add(var_4, var_93);
                    // wrapid = id1                                                               <L 4103>
                    var_95 = wp::copy(var_46);
                    // id1 = wrap_objid[adr + j + 2]                                              <L 4104>
                    var_96 = wp::add(var_8, var_20);
                    var_98 = wp::add(var_96, var_97);
                    var_99 = wp::address(var_wrap_objid, var_98);
                    var_101 = wp::load(var_99);
                    var_100 = wp::copy(var_101);
                    // if wp.norm_l2(wpnt_geom0) < MJ_MAXVAL:                                     <L 4105>
                    var_102 = norm_l2_0(var_92);
                    var_104 = (var_102 < var_103);
                    if (var_104) {
                        // wpnt_geom1 = wp.spatial_bottom(wrap_geom_xpos)                         <L 4106>
                        var_105 = wp::spatial_bottom(var_90);
                        // wpnt_site1 = site_xpos_in[worldid, id1]                                <L 4107>
                        var_106 = wp::address(var_site_xpos_in, var_0, var_100);
                        var_108 = wp::load(var_106);
                        var_107 = wp::copy(var_108);
                        // row0 = (wrapcount + 0) // 2                                            <L 4110>
                        var_110 = wp::add(var_2, var_109);
                        var_112 = wp::floordiv(var_110, var_111);
                        // col0 = (wrapcount + 0) % 2                                             <L 4111>
                        var_114 = wp::add(var_2, var_113);
                        var_116 = wp::mod(var_114, var_115);
                        // row1 = (wrapcount + 1) // 2                                            <L 4112>
                        var_118 = wp::add(var_2, var_117);
                        var_120 = wp::floordiv(var_118, var_119);
                        // col1 = (wrapcount + 1) % 2                                             <L 4113>
                        var_122 = wp::add(var_2, var_121);
                        var_124 = wp::mod(var_122, var_123);
                        // row2 = (wrapcount + 2) // 2                                            <L 4114>
                        var_126 = wp::add(var_2, var_125);
                        var_128 = wp::floordiv(var_126, var_127);
                        // col2 = (wrapcount + 2) % 2                                             <L 4115>
                        var_130 = wp::add(var_2, var_129);
                        var_132 = wp::mod(var_130, var_131);
                        // row3 = (wrapcount + 3) // 2                                            <L 4116>
                        var_134 = wp::add(var_2, var_133);
                        var_136 = wp::floordiv(var_134, var_135);
                        // col3 = (wrapcount + 3) % 2                                             <L 4117>
                        var_138 = wp::add(var_2, var_137);
                        var_140 = wp::mod(var_138, var_139);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]             <L 4119>
                        var_142 = wp::extract(var_82, var_141);
                        var_144 = wp::mul(var_143, var_116);
                        var_146 = wp::add(var_144, var_145);
                        wp::index(var_wrap_xpos_out, var_0, var_112)[var_146] = var_142;
                        // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]             <L 4120>
                        var_148 = wp::extract(var_82, var_147);
                        var_150 = wp::mul(var_149, var_116);
                        var_152 = wp::add(var_150, var_151);
                        wp::index(var_wrap_xpos_out, var_0, var_112)[var_152] = var_148;
                        // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]             <L 4121>
                        var_154 = wp::extract(var_82, var_153);
                        var_156 = wp::mul(var_155, var_116);
                        var_158 = wp::add(var_156, var_157);
                        wp::index(var_wrap_xpos_out, var_0, var_112)[var_158] = var_154;
                        // wrap_xpos_out[worldid, row1][3 * col1 + 0] = wpnt_geom0[0]             <L 4123>
                        var_160 = wp::extract(var_92, var_159);
                        var_162 = wp::mul(var_161, var_124);
                        var_164 = wp::add(var_162, var_163);
                        wp::index(var_wrap_xpos_out, var_0, var_120)[var_164] = var_160;
                        // wrap_xpos_out[worldid, row1][3 * col1 + 1] = wpnt_geom0[1]             <L 4124>
                        var_166 = wp::extract(var_92, var_165);
                        var_168 = wp::mul(var_167, var_124);
                        var_170 = wp::add(var_168, var_169);
                        wp::index(var_wrap_xpos_out, var_0, var_120)[var_170] = var_166;
                        // wrap_xpos_out[worldid, row1][3 * col1 + 2] = wpnt_geom0[2]             <L 4125>
                        var_172 = wp::extract(var_92, var_171);
                        var_174 = wp::mul(var_173, var_124);
                        var_176 = wp::add(var_174, var_175);
                        wp::index(var_wrap_xpos_out, var_0, var_120)[var_176] = var_172;
                        // wrap_xpos_out[worldid, row2][3 * col2 + 0] = wpnt_geom1[0]             <L 4127>
                        var_178 = wp::extract(var_105, var_177);
                        var_180 = wp::mul(var_179, var_132);
                        var_182 = wp::add(var_180, var_181);
                        wp::index(var_wrap_xpos_out, var_0, var_128)[var_182] = var_178;
                        // wrap_xpos_out[worldid, row2][3 * col2 + 1] = wpnt_geom1[1]             <L 4128>
                        var_184 = wp::extract(var_105, var_183);
                        var_186 = wp::mul(var_185, var_132);
                        var_188 = wp::add(var_186, var_187);
                        wp::index(var_wrap_xpos_out, var_0, var_128)[var_188] = var_184;
                        // wrap_xpos_out[worldid, row2][3 * col2 + 2] = wpnt_geom1[2]             <L 4129>
                        var_190 = wp::extract(var_105, var_189);
                        var_192 = wp::mul(var_191, var_132);
                        var_194 = wp::add(var_192, var_193);
                        wp::index(var_wrap_xpos_out, var_0, var_128)[var_194] = var_190;
                        // wrap_xpos_out[worldid, row3][3 * col3 + 0] = wpnt_site1[0]             <L 4131>
                        var_196 = wp::extract(var_107, var_195);
                        var_198 = wp::mul(var_197, var_140);
                        var_200 = wp::add(var_198, var_199);
                        wp::index(var_wrap_xpos_out, var_0, var_136)[var_200] = var_196;
                        // wrap_xpos_out[worldid, row3][3 * col3 + 1] = wpnt_site1[1]             <L 4132>
                        var_202 = wp::extract(var_107, var_201);
                        var_204 = wp::mul(var_203, var_140);
                        var_206 = wp::add(var_204, var_205);
                        wp::index(var_wrap_xpos_out, var_0, var_136)[var_206] = var_202;
                        // wrap_xpos_out[worldid, row3][3 * col3 + 2] = wpnt_site1[2]             <L 4133>
                        var_208 = wp::extract(var_107, var_207);
                        var_210 = wp::mul(var_209, var_140);
                        var_212 = wp::add(var_210, var_211);
                        wp::index(var_wrap_xpos_out, var_0, var_136)[var_212] = var_208;
                        // wrap_obj_out[worldid, row0][col0] = -1                                 <L 4135>
                        wp::index(var_wrap_obj_out, var_0, var_112)[var_116] = var_213;
                        // wrap_obj_out[worldid, row1][col1] = wrapid                             <L 4136>
                        wp::index(var_wrap_obj_out, var_0, var_120)[var_124] = var_95;
                        // wrap_obj_out[worldid, row2][col2] = wrapid                             <L 4137>
                        wp::index(var_wrap_obj_out, var_0, var_128)[var_132] = var_95;
                        // wrapnum += 3                                                           <L 4139>
                        var_215 = wp::add(var_11, var_214);
                        // wrapcount += 3                                                         <L 4140>
                        var_217 = wp::add(var_2, var_216);
                        // j += 2                                                                 <L 4141>
                        var_219 = wp::add(var_20, var_218);
                    }
                    if (!var_104) {
                        // row0 = (wrapcount + 0) // 2                                            <L 4144>
                        var_221 = wp::add(var_2, var_220);
                        var_223 = wp::floordiv(var_221, var_222);
                        // col0 = (wrapcount + 0) % 2                                             <L 4145>
                        var_225 = wp::add(var_2, var_224);
                        var_227 = wp::mod(var_225, var_226);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]             <L 4147>
                        var_229 = wp::extract(var_82, var_228);
                        var_231 = wp::mul(var_230, var_227);
                        var_233 = wp::add(var_231, var_232);
                        wp::index(var_wrap_xpos_out, var_0, var_223)[var_233] = var_229;
                        // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]             <L 4148>
                        var_235 = wp::extract(var_82, var_234);
                        var_237 = wp::mul(var_236, var_227);
                        var_239 = wp::add(var_237, var_238);
                        wp::index(var_wrap_xpos_out, var_0, var_223)[var_239] = var_235;
                        // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]             <L 4149>
                        var_241 = wp::extract(var_82, var_240);
                        var_243 = wp::mul(var_242, var_227);
                        var_245 = wp::add(var_243, var_244);
                        wp::index(var_wrap_xpos_out, var_0, var_223)[var_245] = var_241;
                        // wrap_obj_out[worldid, row0][col0] = -1                                 <L 4151>
                        wp::index(var_wrap_obj_out, var_0, var_223)[var_227] = var_246;
                        // wrapnum += 1                                                           <L 4153>
                        var_248 = wp::add(var_11, var_247);
                        // wrapcount += 1                                                         <L 4154>
                        var_250 = wp::add(var_2, var_249);
                        // j += 2                                                                 <L 4155>
                        var_252 = wp::add(var_20, var_251);
                    }
                    var_253 = wp::where(var_104, var_217, var_250);
                    var_254 = wp::where(var_104, var_215, var_248);
                    var_255 = wp::where(var_104, var_219, var_252);
                    var_256 = wp::where(var_104, var_112, var_223);
                    var_257 = wp::where(var_104, var_116, var_227);
                }
                if (!var_84) {
                    // row0 = (wrapcount + 0) // 2                                                <L 4158>
                    var_259 = wp::add(var_2, var_258);
                    var_261 = wp::floordiv(var_259, var_260);
                    // col0 = (wrapcount + 0) % 2                                                 <L 4159>
                    var_263 = wp::add(var_2, var_262);
                    var_265 = wp::mod(var_263, var_264);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]                 <L 4161>
                    var_267 = wp::extract(var_82, var_266);
                    var_269 = wp::mul(var_268, var_265);
                    var_271 = wp::add(var_269, var_270);
                    wp::index(var_wrap_xpos_out, var_0, var_261)[var_271] = var_267;
                    // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]                 <L 4162>
                    var_273 = wp::extract(var_82, var_272);
                    var_275 = wp::mul(var_274, var_265);
                    var_277 = wp::add(var_275, var_276);
                    wp::index(var_wrap_xpos_out, var_0, var_261)[var_277] = var_273;
                    // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]                 <L 4163>
                    var_279 = wp::extract(var_82, var_278);
                    var_281 = wp::mul(var_280, var_265);
                    var_283 = wp::add(var_281, var_282);
                    wp::index(var_wrap_xpos_out, var_0, var_261)[var_283] = var_279;
                    // wrap_obj_out[worldid, row0][col0] = -1                                     <L 4165>
                    wp::index(var_wrap_obj_out, var_0, var_261)[var_265] = var_284;
                    // wrapnum += 1                                                               <L 4167>
                    var_286 = wp::add(var_11, var_285);
                    // wrapcount += 1                                                             <L 4168>
                    var_288 = wp::add(var_2, var_287);
                    // j += 1                                                                     <L 4169>
                    var_290 = wp::add(var_20, var_289);
                }
                var_291 = wp::where(var_84, var_253, var_288);
                var_292 = wp::where(var_84, var_94, var_4);
                var_293 = wp::where(var_84, var_254, var_286);
                var_294 = wp::where(var_84, var_255, var_290);
                var_295 = wp::where(var_84, var_100, var_46);
                var_296 = wp::where(var_84, var_256, var_261);
                var_297 = wp::where(var_84, var_257, var_265);
                // if adr + j + 1 < wrap_type.shape[0]:                                           <L 4172>
                var_298 = wp::add(var_8, var_294);
                var_300 = wp::add(var_298, var_299);
                var_301 = &(var_wrap_type.shape);
                var_304 = wp::load(var_301);
                var_303 = wp::extract(var_304, var_302);
                var_305 = (var_300 < var_303);
                if (var_305) {
                    // last_before_pulley = wrap_type[adr + j + 1] == WrapType.PULLEY             <L 4173>
                    var_306 = wp::add(var_8, var_294);
                    var_308 = wp::add(var_306, var_307);
                    var_309 = wp::address(var_wrap_type, var_308);
                    var_312 = wp::load(var_309);
                    var_311 = (var_312 == var_310);
                }
                if (!var_305) {
                    // last_before_pulley = False                                                 <L 4175>
                }
                var_314 = wp::where(var_305, var_311, var_313);
                // if j == tendonnum - 1 or last_before_pulley:                                   <L 4177>
                var_317 = wp::sub(var_13, var_316);
                var_318 = (var_294 == var_317);
                var_315 = var_318;
                if (!var_315) {
                    var_315 = var_315 || var_314;
                }
                if (var_315) {
                    // row0 = (wrapcount + 0) // 2                                                <L 4178>
                    var_320 = wp::add(var_291, var_319);
                    var_322 = wp::floordiv(var_320, var_321);
                    // col0 = (wrapcount + 0) % 2                                                 <L 4179>
                    var_324 = wp::add(var_291, var_323);
                    var_326 = wp::mod(var_324, var_325);
                    // wpnt_site1 = site_xpos_in[worldid, id1]                                    <L 4181>
                    var_327 = wp::address(var_site_xpos_in, var_0, var_295);
                    var_329 = wp::load(var_327);
                    var_328 = wp::copy(var_329);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site1[0]                 <L 4182>
                    var_331 = wp::extract(var_328, var_330);
                    var_333 = wp::mul(var_332, var_326);
                    var_335 = wp::add(var_333, var_334);
                    wp::index(var_wrap_xpos_out, var_0, var_322)[var_335] = var_331;
                    // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site1[1]                 <L 4183>
                    var_337 = wp::extract(var_328, var_336);
                    var_339 = wp::mul(var_338, var_326);
                    var_341 = wp::add(var_339, var_340);
                    wp::index(var_wrap_xpos_out, var_0, var_322)[var_341] = var_337;
                    // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site1[2]                 <L 4184>
                    var_343 = wp::extract(var_328, var_342);
                    var_345 = wp::mul(var_344, var_326);
                    var_347 = wp::add(var_345, var_346);
                    wp::index(var_wrap_xpos_out, var_0, var_322)[var_347] = var_343;
                    // wrap_obj_out[worldid, row0][col0] = -1                                     <L 4186>
                    wp::index(var_wrap_obj_out, var_0, var_322)[var_326] = var_348;
                    // wrapnum += 1                                                               <L 4187>
                    var_350 = wp::add(var_293, var_349);
                    // wrapcount += 1                                                             <L 4188>
                    var_352 = wp::add(var_291, var_351);
                }
                var_353 = wp::where(var_315, var_352, var_291);
                var_354 = wp::where(var_315, var_350, var_293);
                var_355 = wp::where(var_315, var_328, var_107);
                var_356 = wp::where(var_315, var_322, var_296);
                var_357 = wp::where(var_315, var_326, var_297);
                wp::assign(var_2, var_353);
                wp::assign(var_4, var_292);
                wp::assign(var_11, var_354);
                wp::assign(var_20, var_294);
        goto start_while_2;
        end_while_2:;
            // ten_wrapnum_out[worldid, i] = wrapnum                                              <L 4190>
            wp::array_store(var_ten_wrapnum_out, var_0, var_6, var_11);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _qLDiag_div_3444613a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_D_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        const wp::float32 var_9 = 1.0;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        //---------
        // forward
        // def _qLDiag_div(                                                                       <L 1210>
        // worldid, dofid = wp.tid()                                                              <L 1219>
        builtin_tid2d(var_0, var_1);
        // diag_i = M_rowadr[dofid] + M_rownnz[dofid] - 1  # Address of diagonal element of i       <L 1220>
        var_2 = wp::address(var_M_rowadr, var_1);
        var_3 = wp::address(var_M_rownnz, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        var_8 = wp::sub(var_4, var_7);
        // D_out[worldid, dofid] = 1.0 / L_in[worldid, diag_i]                                    <L 1221>
        var_10 = wp::address(var_L_in, var_0, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::div(var_9, var_12);
        wp::array_store(var_D_out, var_0, var_1, var_11);
    }
}



extern "C" __global__ void _cfrc_ext_equality_3475e5f6_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_pos,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::int32> var_ne_connect_in,
    wp::array_t<wp::int32> var_ne_weld_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 3;
        wp::int32 var_9;
        const wp::int32 var_10 = 6;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        const wp::int32 var_15 = 3;
        wp::int32 var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::int32 var_21 = 6;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 3;
        wp::int32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 4;
        wp::int32 var_28;
        wp::float32* var_29;
        const wp::int32 var_30 = 5;
        wp::int32 var_31;
        wp::float32* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::int32 var_37;
        wp::vec_t<3, wp::float32> var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::float32* var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::float32* var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::shape_t* var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        wp::shape_t var_58;
        wp::int32 var_59;
        wp::vec_t<11, wp::float32>* var_60;
        wp::vec_t<11, wp::float32> var_61;
        wp::vec_t<11, wp::float32> var_62;
        wp::int32* var_63;
        const wp::int32 var_64 = 1;
        bool var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32* var_70;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 2;
        wp::float32 var_88;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 3;
        wp::float32 var_91;
        const wp::int32 var_92 = 4;
        wp::float32 var_93;
        const wp::int32 var_94 = 5;
        wp::float32 var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::shape_t* var_98;
        const wp::int32 var_99 = 0;
        wp::int32 var_100;
        wp::shape_t var_101;
        wp::int32 var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::mat_t<3, 3, wp::float32>* var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::mat_t<3, 3, wp::float32> var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::int32* var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::vec_t<6, wp::float32> var_121;
        wp::array_t<wp::vec_t<6, wp::float32>> var_122;
        wp::vec_t<6, wp::float32> var_123;
        const wp::int32 var_124 = 3;
        wp::float32 var_125;
        const wp::int32 var_126 = 4;
        wp::float32 var_127;
        const wp::int32 var_128 = 5;
        wp::float32 var_129;
        wp::vec_t<3, wp::float32> var_130;
        const wp::int32 var_131 = 0;
        wp::float32 var_132;
        const wp::int32 var_133 = 1;
        wp::float32 var_134;
        const wp::int32 var_135 = 2;
        wp::float32 var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::shape_t* var_139;
        const wp::int32 var_140 = 0;
        wp::int32 var_141;
        wp::shape_t var_142;
        wp::int32 var_143;
        wp::vec_t<3, wp::float32>* var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::mat_t<3, 3, wp::float32>* var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::mat_t<3, 3, wp::float32> var_150;
        wp::vec_t<3, wp::float32>* var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::int32* var_154;
        wp::vec_t<3, wp::float32>* var_155;
        wp::int32 var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::vec_t<3, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<6, wp::float32> var_162;
        wp::array_t<wp::vec_t<6, wp::float32>> var_163;
        wp::vec_t<6, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::vec_t<3, wp::float32> var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::vec_t<6, wp::float32> var_169;
        //---------
        // forward
        // def _cfrc_ext_equality(                                                                <L 1570>
        // worldid, eqid = wp.tid()                                                               <L 1591>
        builtin_tid2d(var_0, var_1);
        // ne_connect = ne_connect_in[worldid]                                                    <L 1593>
        var_2 = wp::address(var_ne_connect_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // ne_weld = ne_weld_in[worldid]                                                          <L 1594>
        var_5 = wp::address(var_ne_weld_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // num_connect = ne_connect // 3                                                          <L 1595>
        var_9 = wp::floordiv(var_3, var_8);
        // if eqid >= num_connect + ne_weld // 6:                                                 <L 1597>
        var_11 = wp::floordiv(var_6, var_10);
        var_12 = wp::add(var_9, var_11);
        var_13 = (var_1 >= var_12);
        if (var_13) {
            // return                                                                             <L 1598>
            continue;
        }
        // is_connect = eqid < num_connect                                                        <L 1600>
        var_14 = (var_1 < var_9);
        // if is_connect:                                                                         <L 1601>
        if (var_14) {
            // efcid = 3 * eqid                                                                   <L 1602>
            var_16 = wp::mul(var_15, var_1);
            // cfrc_torque = wp.vec3(0.0, 0.0, 0.0)  # no torque from connect                     <L 1603>
            var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        }
        if (!var_14) {
            // efcid = 6 * eqid - ne_connect                                                      <L 1605>
            var_22 = wp::mul(var_21, var_1);
            var_23 = wp::sub(var_22, var_3);
            // cfrc_torque = wp.vec3(efc_force_in[worldid, efcid + 3], efc_force_in[worldid, efcid + 4], efc_force_in[worldid, efcid + 5])       <L 1606>
            var_25 = wp::add(var_23, var_24);
            var_26 = wp::address(var_efc_force_in, var_0, var_25);
            var_28 = wp::add(var_23, var_27);
            var_29 = wp::address(var_efc_force_in, var_0, var_28);
            var_31 = wp::add(var_23, var_30);
            var_32 = wp::address(var_efc_force_in, var_0, var_31);
            var_34 = wp::load(var_26);
            var_35 = wp::load(var_29);
            var_36 = wp::load(var_32);
            var_33 = wp::vec_t<3, wp::float32>(var_34, var_35, var_36);
        }
        var_37 = wp::where(var_14, var_16, var_23);
        var_38 = wp::where(var_14, var_20, var_33);
        // cfrc_force = wp.vec3(                                                                  <L 1608>
        // efc_force_in[worldid, efcid + 0],                                                      <L 1609>
        var_40 = wp::add(var_37, var_39);
        var_41 = wp::address(var_efc_force_in, var_0, var_40);
        // efc_force_in[worldid, efcid + 1],                                                      <L 1610>
        var_43 = wp::add(var_37, var_42);
        var_44 = wp::address(var_efc_force_in, var_0, var_43);
        // efc_force_in[worldid, efcid + 2],                                                      <L 1611>
        var_46 = wp::add(var_37, var_45);
        var_47 = wp::address(var_efc_force_in, var_0, var_46);
        var_49 = wp::load(var_41);
        var_50 = wp::load(var_44);
        var_51 = wp::load(var_47);
        var_48 = wp::vec_t<3, wp::float32>(var_49, var_50, var_51);
        // id = efc_id_in[worldid, efcid]                                                         <L 1614>
        var_52 = wp::address(var_efc_id_in, var_0, var_37);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // eq_data_ = eq_data[worldid % eq_data.shape[0], id]                                     <L 1615>
        var_55 = &(var_eq_data.shape);
        var_58 = wp::load(var_55);
        var_57 = wp::extract(var_58, var_56);
        var_59 = wp::mod(var_0, var_57);
        var_60 = wp::address(var_eq_data, var_59, var_53);
        var_62 = wp::load(var_60);
        var_61 = wp::copy(var_62);
        // body_semantic = eq_objtype[id] == ObjType.BODY                                         <L 1616>
        var_63 = wp::address(var_eq_objtype, var_53);
        var_66 = wp::load(var_63);
        var_65 = (var_66 == var_64);
        // obj1 = eq_obj1id[id]                                                                   <L 1618>
        var_67 = wp::address(var_eq_obj1id, var_53);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // obj2 = eq_obj2id[id]                                                                   <L 1619>
        var_70 = wp::address(var_eq_obj2id, var_53);
        var_72 = wp::load(var_70);
        var_71 = wp::copy(var_72);
        // if body_semantic:                                                                      <L 1621>
        if (var_65) {
            // bodyid1 = obj1                                                                     <L 1622>
            var_73 = wp::copy(var_68);
            // bodyid2 = obj2                                                                     <L 1623>
            var_74 = wp::copy(var_71);
        }
        if (!var_65) {
            // bodyid1 = site_bodyid[obj1]                                                        <L 1625>
            var_75 = wp::address(var_site_bodyid, var_68);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // bodyid2 = site_bodyid[obj2]                                                        <L 1626>
            var_78 = wp::address(var_site_bodyid, var_71);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
        }
        var_81 = wp::where(var_65, var_73, var_76);
        var_82 = wp::where(var_65, var_74, var_79);
        // if bodyid1:                                                                            <L 1629>
        if (var_81) {
            // if body_semantic:                                                                  <L 1630>
            if (var_65) {
                // if is_connect:                                                                 <L 1631>
                if (var_14) {
                    // offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])                    <L 1632>
                    var_84 = wp::extract(var_61, var_83);
                    var_86 = wp::extract(var_61, var_85);
                    var_88 = wp::extract(var_61, var_87);
                    var_89 = wp::vec_t<3, wp::float32>(var_84, var_86, var_88);
                }
                if (!var_14) {
                    // offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])                    <L 1634>
                    var_91 = wp::extract(var_61, var_90);
                    var_93 = wp::extract(var_61, var_92);
                    var_95 = wp::extract(var_61, var_94);
                    var_96 = wp::vec_t<3, wp::float32>(var_91, var_93, var_95);
                }
                var_97 = wp::where(var_14, var_89, var_96);
            }
            if (!var_65) {
                // offset = site_pos[worldid % site_pos.shape[0], obj1]                           <L 1636>
                var_98 = &(var_site_pos.shape);
                var_101 = wp::load(var_98);
                var_100 = wp::extract(var_101, var_99);
                var_102 = wp::mod(var_0, var_100);
                var_103 = wp::address(var_site_pos, var_102, var_68);
                var_105 = wp::load(var_103);
                var_104 = wp::copy(var_105);
            }
            var_106 = wp::where(var_65, var_97, var_104);
            // pos = xmat_in[worldid, bodyid1] @ offset + xpos_in[worldid, bodyid1]               <L 1639>
            var_107 = wp::address(var_xmat_in, var_0, var_81);
            var_109 = wp::load(var_107);
            var_108 = wp::mul(var_109, var_106);
            var_110 = wp::address(var_xpos_in, var_0, var_81);
            var_112 = wp::load(var_110);
            var_111 = wp::add(var_108, var_112);
            // newpos = subtree_com_in[worldid, body_rootid[bodyid1]]                             <L 1642>
            var_113 = wp::address(var_body_rootid, var_81);
            var_115 = wp::load(var_113);
            var_114 = wp::address(var_subtree_com_in, var_0, var_115);
            var_117 = wp::load(var_114);
            var_116 = wp::copy(var_117);
            // dif = newpos - pos                                                                 <L 1644>
            var_118 = wp::sub(var_116, var_111);
            // cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)       <L 1645>
            var_119 = wp::cross(var_118, var_48);
            var_120 = wp::sub(var_38, var_119);
            var_121 = wp::vec_t<6, wp::float32>(var_120, var_48);
            // wp.atomic_add(cfrc_ext_out[worldid], bodyid1, cfrc_com)                            <L 1648>
            var_122 = wp::view(var_cfrc_ext_out, var_0);
            var_123 = wp::atomic_add(var_122, var_81, var_121);
        }
        // if bodyid2:                                                                            <L 1651>
        if (var_82) {
            // if body_semantic:                                                                  <L 1652>
            if (var_65) {
                // if is_connect:                                                                 <L 1653>
                if (var_14) {
                    // offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])                    <L 1654>
                    var_125 = wp::extract(var_61, var_124);
                    var_127 = wp::extract(var_61, var_126);
                    var_129 = wp::extract(var_61, var_128);
                    var_130 = wp::vec_t<3, wp::float32>(var_125, var_127, var_129);
                }
                if (!var_14) {
                    // offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])                    <L 1656>
                    var_132 = wp::extract(var_61, var_131);
                    var_134 = wp::extract(var_61, var_133);
                    var_136 = wp::extract(var_61, var_135);
                    var_137 = wp::vec_t<3, wp::float32>(var_132, var_134, var_136);
                }
                var_138 = wp::where(var_14, var_130, var_137);
            }
            if (!var_65) {
                // offset = site_pos[worldid % site_pos.shape[0], obj2]                           <L 1658>
                var_139 = &(var_site_pos.shape);
                var_142 = wp::load(var_139);
                var_141 = wp::extract(var_142, var_140);
                var_143 = wp::mod(var_0, var_141);
                var_144 = wp::address(var_site_pos, var_143, var_71);
                var_146 = wp::load(var_144);
                var_145 = wp::copy(var_146);
            }
            var_147 = wp::where(var_65, var_138, var_145);
            // pos = xmat_in[worldid, bodyid2] @ offset + xpos_in[worldid, bodyid2]               <L 1661>
            var_148 = wp::address(var_xmat_in, var_0, var_82);
            var_150 = wp::load(var_148);
            var_149 = wp::mul(var_150, var_147);
            var_151 = wp::address(var_xpos_in, var_0, var_82);
            var_153 = wp::load(var_151);
            var_152 = wp::add(var_149, var_153);
            // newpos = subtree_com_in[worldid, body_rootid[bodyid2]]                             <L 1664>
            var_154 = wp::address(var_body_rootid, var_82);
            var_156 = wp::load(var_154);
            var_155 = wp::address(var_subtree_com_in, var_0, var_156);
            var_158 = wp::load(var_155);
            var_157 = wp::copy(var_158);
            // dif = newpos - pos                                                                 <L 1666>
            var_159 = wp::sub(var_157, var_152);
            // cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)       <L 1667>
            var_160 = wp::cross(var_159, var_48);
            var_161 = wp::sub(var_38, var_160);
            var_162 = wp::vec_t<6, wp::float32>(var_161, var_48);
            // wp.atomic_sub(cfrc_ext_out[worldid], bodyid2, cfrc_com)                            <L 1670>
            var_163 = wp::view(var_cfrc_ext_out, var_0);
            var_164 = wp::atomic_sub(var_163, var_82, var_162);
        }
        var_165 = wp::where(var_82, var_147, var_106);
        var_166 = wp::where(var_82, var_152, var_111);
        var_167 = wp::where(var_82, var_157, var_116);
        var_168 = wp::where(var_82, var_159, var_118);
        var_169 = wp::where(var_82, var_162, var_121);
    }
}



extern "C" __global__ void _subtree_com_acc_350f0560_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        //---------
        // forward
        // def _subtree_com_acc(                                                                  <L 700>
        // worldid, nodeid = wp.tid()                                                             <L 710>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 711>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 712>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if bodyid != 0:                                                                        <L 713>
        var_9 = (var_3 != var_8);
        if (var_9) {
            // wp.atomic_add(subtree_com_out, worldid, pid, subtree_com_in[worldid, bodyid])       <L 714>
            var_10 = wp::address(var_subtree_com_in, var_0, var_3);
            var_12 = wp::load(var_10);
            var_11 = wp::atomic_add(var_subtree_com_out, var_0, var_6, var_12);
        }
    }
}



extern "C" __global__ void _angular_momentum_35b41762_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_subtree_bodyvel_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<6, wp::float32>* var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::shape_t* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::shape_t var_31;
        wp::int32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::shape_t* var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::shape_t var_39;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::array_t<wp::vec_t<3, wp::float32>> var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::array_t<wp::vec_t<3, wp::float32>> var_58;
        wp::vec_t<3, wp::float32> var_59;
        //---------
        // forward
        // def _angular_momentum(                                                                 <L 3559>
        // worldid, nodeid = wp.tid()                                                             <L 3574>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 3575>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if bodyid == 0:                                                                        <L 3577>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 3578>
            continue;
        }
        // pid = body_parentid[bodyid]                                                            <L 3580>
        var_7 = wp::address(var_body_parentid, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 3582>
        var_10 = wp::address(var_xipos_in, var_0, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // com = subtree_com_in[worldid, bodyid]                                                  <L 3583>
        var_13 = wp::address(var_subtree_com_in, var_0, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // com_parent = subtree_com_in[worldid, pid]                                              <L 3584>
        var_16 = wp::address(var_subtree_com_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // vel = subtree_bodyvel_in[worldid, bodyid]                                              <L 3585>
        var_19 = wp::address(var_subtree_bodyvel_in, var_0, var_3);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // linvel = subtree_linvel_in[worldid, bodyid]                                            <L 3586>
        var_22 = wp::address(var_subtree_linvel_in, var_0, var_3);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // linvel_parent = subtree_linvel_in[worldid, pid]  # Data field                          <L 3587>
        var_25 = wp::address(var_subtree_linvel_in, var_0, var_8);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 3588>
        var_28 = &(var_body_mass.shape);
        var_31 = wp::load(var_28);
        var_30 = wp::extract(var_31, var_29);
        var_32 = wp::mod(var_0, var_30);
        var_33 = wp::address(var_body_mass, var_32, var_3);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // subtreemass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]            <L 3589>
        var_36 = &(var_body_subtreemass.shape);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_40 = wp::mod(var_0, var_38);
        var_41 = wp::address(var_body_subtreemass, var_40, var_3);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // dx = xipos - com                                                                       <L 3592>
        var_44 = wp::sub(var_11, var_14);
        // dv = wp.spatial_bottom(vel) - linvel                                                   <L 3593>
        var_45 = wp::spatial_bottom(var_20);
        var_46 = wp::sub(var_45, var_23);
        // dp = dv * mass                                                                         <L 3594>
        var_47 = wp::mul(var_46, var_34);
        // dL = wp.cross(dx, dp)                                                                  <L 3595>
        var_48 = wp::cross(var_44, var_47);
        // subtree_angmom_out[worldid, bodyid] += dL                                              <L 3598>
        var_49 = wp::atomic_add(var_subtree_angmom_out, var_0, var_3, var_48);
        // wp.atomic_add(subtree_angmom_out[worldid], pid, subtree_angmom_out[worldid, bodyid])       <L 3601>
        var_50 = wp::view(var_subtree_angmom_out, var_0);
        var_51 = wp::address(var_subtree_angmom_out, var_0, var_3);
        var_53 = wp::load(var_51);
        var_52 = wp::atomic_add(var_50, var_8, var_53);
        // dx = com - com_parent                                                                  <L 3604>
        var_54 = wp::sub(var_14, var_17);
        // dv = linvel - linvel_parent                                                            <L 3605>
        var_55 = wp::sub(var_23, var_26);
        // dv *= subtreemass                                                                      <L 3606>
        var_56 = wp::mul(var_55, var_42);
        // dL = wp.cross(dx, dv)                                                                  <L 3607>
        var_57 = wp::cross(var_54, var_56);
        // wp.atomic_add(subtree_angmom_out[worldid], pid, dL)                                    <L 3608>
        var_58 = wp::view(var_subtree_angmom_out, var_0);
        var_59 = wp::atomic_add(var_58, var_8, var_57);
    }
}



extern "C" __global__ void _joint_tendon_35c34b14_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::int32> var_tendon_jnt_adr,
    wp::array_t<wp::int32> var_wrap_jnt_adr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::int32* var_14;
        wp::float32* var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::array_t<wp::float32> var_19;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        bool var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        //---------
        // forward
        // def _joint_tendon(                                                                     <L 3655>
        // worldid, wrapid = wp.tid()                                                             <L 3672>
        builtin_tid2d(var_0, var_1);
        // tenid = tendon_jnt_adr[wrapid]                                                         <L 3674>
        var_2 = wp::address(var_tendon_jnt_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // wrapjntid = wrap_jnt_adr[wrapid]                                                       <L 3675>
        var_5 = wp::address(var_wrap_jnt_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // wrapobjid = wrap_objid[wrapjntid]                                                      <L 3676>
        var_8 = wp::address(var_wrap_objid, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // prm = wrap_prm[wrapjntid]                                                              <L 3677>
        var_11 = wp::address(var_wrap_prm, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // L = prm * qpos_in[worldid, jnt_qposadr[wrapobjid]]                                     <L 3680>
        var_14 = wp::address(var_jnt_qposadr, var_9);
        var_16 = wp::load(var_14);
        var_15 = wp::address(var_qpos_in, var_0, var_16);
        var_18 = wp::load(var_15);
        var_17 = wp::mul(var_12, var_18);
        // wp.atomic_add(ten_length_out[worldid], tenid, L)                                       <L 3681>
        var_19 = wp::view(var_ten_length_out, var_0);
        var_20 = wp::atomic_add(var_19, var_3, var_17);
        // dofadr = jnt_dofadr[wrapobjid]                                                         <L 3684>
        var_21 = wp::address(var_jnt_dofadr, var_9);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 3685>
        var_24 = wp::address(var_ten_J_rowadr, var_3);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 3686>
        var_27 = wp::address(var_ten_J_rownnz, var_3);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // for k in range(rownnz):                                                                <L 3687>
        var_30 = wp::range(var_28);
        start_for_0:;
            if (iter_cmp(var_30) == 0) goto end_for_0;
            var_31 = wp::iter_next(var_30);
            // if ten_J_colind[rowadr + k] == dofadr:                                             <L 3688>
            var_32 = wp::add(var_25, var_31);
            var_33 = wp::address(var_ten_J_colind, var_32);
            var_35 = wp::load(var_33);
            var_34 = (var_35 == var_22);
            if (var_34) {
                // ten_J_out[worldid, rowadr + k] = prm                                           <L 3689>
                var_36 = wp::add(var_25, var_31);
                wp::array_store(var_ten_J_out, var_0, var_36, var_12);
                // break                                                                          <L 3690>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _cfrc_3922b605_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_in,
    bool var_flg_cfrc_ext,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 0.0;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<6, wp::float32> var_10;
        const wp::int32 var_11 = 0;
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::vec_t<10, wp::float32>* var_15;
        wp::vec_t<10, wp::float32> var_16;
        wp::vec_t<10, wp::float32> var_17;
        wp::vec_t<6, wp::float32>* var_18;
        wp::vec_t<6, wp::float32> var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::vec_t<6, wp::float32> var_23;
        wp::vec_t<6, wp::float32> var_24;
        wp::vec_t<6, wp::float32>* var_25;
        wp::vec_t<6, wp::float32> var_26;
        wp::vec_t<6, wp::float32> var_27;
        wp::vec_t<6, wp::float32> var_28;
        //---------
        // forward
        // def _cfrc(                                                                             <L 1429>
        // worldid, bodyid = wp.tid()                                                             <L 1440>
        builtin_tid2d(var_0, var_1);
        // if bodyid == 0:                                                                        <L 1441>
        var_3 = (var_1 == var_2);
        if (var_3) {
            // cfrc_int_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 1442>
            var_10 = wp::vec_t<6, wp::float32>({var_4, var_5, var_6, var_7, var_8, var_9});
            wp::array_store(var_cfrc_int_out, var_0, var_11, var_10);
            // return                                                                             <L 1443>
            continue;
        }
        // cacc = cacc_in[worldid, bodyid]                                                        <L 1444>
        var_12 = wp::address(var_cacc_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // cinert = cinert_in[worldid, bodyid]                                                    <L 1445>
        var_15 = wp::address(var_cinert_in, var_0, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 1446>
        var_18 = wp::address(var_cvel_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // frc = math.inert_vec(cinert, cacc)                                                     <L 1447>
        var_21 = inert_vec_0(var_16, var_13);
        // frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))                     <L 1448>
        var_22 = inert_vec_0(var_16, var_19);
        var_23 = motion_cross_force_0(var_19, var_22);
        var_24 = wp::add(var_21, var_23);
        // if flg_cfrc_ext:                                                                       <L 1449>
        if (var_flg_cfrc_ext) {
            // frc -= cfrc_ext_in[worldid, bodyid]                                                <L 1450>
            var_25 = wp::address(var_cfrc_ext_in, var_0, var_1);
            var_27 = wp::load(var_25);
            var_26 = wp::sub(var_24, var_27);
        }
        var_28 = wp::where(var_flg_cfrc_ext, var_26, var_24);
        // cfrc_int_out[worldid, bodyid] = frc                                                    <L 1452>
        wp::array_store(var_cfrc_int_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _cdof_3e73eb78_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_jnt_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xanchor_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xaxis_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::mat_t<3, 3, wp::float32>* var_14;
        wp::mat_t<3, 3, wp::float32> var_15;
        wp::mat_t<3, 3, wp::float32> var_16;
        wp::int32* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::int32 var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::array_t<wp::vec_t<6, wp::float32>> var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 0.0;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 1.0;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        wp::vec_t<6, wp::float32> var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        const wp::float32 var_36 = 0.0;
        const wp::float32 var_37 = 0.0;
        const wp::float32 var_38 = 0.0;
        const wp::float32 var_39 = 0.0;
        const wp::float32 var_40 = 1.0;
        const wp::float32 var_41 = 0.0;
        wp::vec_t<6, wp::float32> var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        const wp::float32 var_45 = 0.0;
        const wp::float32 var_46 = 0.0;
        const wp::float32 var_47 = 0.0;
        const wp::float32 var_48 = 0.0;
        const wp::float32 var_49 = 0.0;
        const wp::float32 var_50 = 1.0;
        wp::vec_t<6, wp::float32> var_51;
        const wp::int32 var_52 = 2;
        wp::int32 var_53;
        const wp::int32 var_54 = 0;
        wp::vec_t<3, wp::float32> var_55;
        const wp::int32 var_56 = 0;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<6, wp::float32> var_59;
        const wp::int32 var_60 = 3;
        wp::int32 var_61;
        const wp::int32 var_62 = 1;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 1;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<6, wp::float32> var_67;
        const wp::int32 var_68 = 4;
        wp::int32 var_69;
        const wp::int32 var_70 = 2;
        wp::vec_t<3, wp::float32> var_71;
        const wp::int32 var_72 = 2;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<6, wp::float32> var_75;
        const wp::int32 var_76 = 5;
        wp::int32 var_77;
        const wp::int32 var_78 = 1;
        bool var_79;
        const wp::int32 var_80 = 0;
        wp::vec_t<3, wp::float32> var_81;
        const wp::int32 var_82 = 0;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<6, wp::float32> var_85;
        const wp::int32 var_86 = 0;
        wp::int32 var_87;
        const wp::int32 var_88 = 1;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 1;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<6, wp::float32> var_93;
        const wp::int32 var_94 = 1;
        wp::int32 var_95;
        const wp::int32 var_96 = 2;
        wp::vec_t<3, wp::float32> var_97;
        const wp::int32 var_98 = 2;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<6, wp::float32> var_101;
        const wp::int32 var_102 = 2;
        wp::int32 var_103;
        const wp::int32 var_104 = 2;
        bool var_105;
        const wp::float32 var_106 = 0.0;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<6, wp::float32> var_108;
        const wp::int32 var_109 = 3;
        bool var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<6, wp::float32> var_112;
        //---------
        // forward
        // def _cdof(                                                                             <L 780>
        // worldid, jntid = wp.tid()                                                              <L 794>
        builtin_tid2d(var_0, var_1);
        // bodyid = jnt_bodyid[jntid]                                                             <L 795>
        var_2 = wp::address(var_jnt_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofid = jnt_dofadr[jntid]                                                              <L 796>
        var_5 = wp::address(var_jnt_dofadr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // jnt_type_ = jnt_type[jntid]                                                            <L 797>
        var_8 = wp::address(var_jnt_type, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // xaxis = xaxis_in[worldid, jntid]                                                       <L 798>
        var_11 = wp::address(var_xaxis_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // xmat = wp.transpose(xmat_in[worldid, bodyid])                                          <L 799>
        var_14 = wp::address(var_xmat_in, var_0, var_3);
        var_16 = wp::load(var_14);
        var_15 = wp::transpose(var_16);
        // offset = subtree_com_in[worldid, body_rootid[bodyid]] - xanchor_in[worldid, jntid]       <L 802>
        var_17 = wp::address(var_body_rootid, var_3);
        var_19 = wp::load(var_17);
        var_18 = wp::address(var_subtree_com_in, var_0, var_19);
        var_20 = wp::address(var_xanchor_in, var_0, var_1);
        var_22 = wp::load(var_18);
        var_23 = wp::load(var_20);
        var_21 = wp::sub(var_22, var_23);
        // res = cdof_out[worldid]                                                                <L 804>
        var_24 = wp::view(var_cdof_out, var_0);
        // if jnt_type_ == JointType.FREE:                                                        <L 805>
        var_26 = (var_9 == var_25);
        if (var_26) {
            // res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)                   <L 806>
            var_33 = wp::vec_t<6, wp::float32>({var_27, var_28, var_29, var_30, var_31, var_32});
            var_35 = wp::add(var_6, var_34);
            wp::array_store(var_24, var_35, var_33);
            // res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)                   <L 807>
            var_42 = wp::vec_t<6, wp::float32>({var_36, var_37, var_38, var_39, var_40, var_41});
            var_44 = wp::add(var_6, var_43);
            wp::array_store(var_24, var_44, var_42);
            // res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)                   <L 808>
            var_51 = wp::vec_t<6, wp::float32>({var_45, var_46, var_47, var_48, var_49, var_50});
            var_53 = wp::add(var_6, var_52);
            wp::array_store(var_24, var_53, var_51);
            // res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))             <L 810>
            var_55 = wp::extract(var_15, var_54);
            var_57 = wp::extract(var_15, var_56);
            var_58 = wp::cross(var_57, var_21);
            var_59 = wp::vec_t<6, wp::float32>(var_55, var_58);
            var_61 = wp::add(var_6, var_60);
            wp::array_store(var_24, var_61, var_59);
            // res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))             <L 811>
            var_63 = wp::extract(var_15, var_62);
            var_65 = wp::extract(var_15, var_64);
            var_66 = wp::cross(var_65, var_21);
            var_67 = wp::vec_t<6, wp::float32>(var_63, var_66);
            var_69 = wp::add(var_6, var_68);
            wp::array_store(var_24, var_69, var_67);
            // res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))             <L 812>
            var_71 = wp::extract(var_15, var_70);
            var_73 = wp::extract(var_15, var_72);
            var_74 = wp::cross(var_73, var_21);
            var_75 = wp::vec_t<6, wp::float32>(var_71, var_74);
            var_77 = wp::add(var_6, var_76);
            wp::array_store(var_24, var_77, var_75);
        }
        if (!var_26) {
            // elif jnt_type_ == JointType.BALL:  # ball                                          <L 813>
            var_79 = (var_9 == var_78);
            if (var_79) {
                // res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))         <L 815>
                var_81 = wp::extract(var_15, var_80);
                var_83 = wp::extract(var_15, var_82);
                var_84 = wp::cross(var_83, var_21);
                var_85 = wp::vec_t<6, wp::float32>(var_81, var_84);
                var_87 = wp::add(var_6, var_86);
                wp::array_store(var_24, var_87, var_85);
                // res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))         <L 816>
                var_89 = wp::extract(var_15, var_88);
                var_91 = wp::extract(var_15, var_90);
                var_92 = wp::cross(var_91, var_21);
                var_93 = wp::vec_t<6, wp::float32>(var_89, var_92);
                var_95 = wp::add(var_6, var_94);
                wp::array_store(var_24, var_95, var_93);
                // res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))         <L 817>
                var_97 = wp::extract(var_15, var_96);
                var_99 = wp::extract(var_15, var_98);
                var_100 = wp::cross(var_99, var_21);
                var_101 = wp::vec_t<6, wp::float32>(var_97, var_100);
                var_103 = wp::add(var_6, var_102);
                wp::array_store(var_24, var_103, var_101);
            }
            if (!var_79) {
                // elif jnt_type_ == JointType.SLIDE:                                             <L 818>
                var_105 = (var_9 == var_104);
                if (var_105) {
                    // res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)                        <L 819>
                    var_107 = wp::vec_t<3, wp::float32>(var_106);
                    var_108 = wp::vec_t<6, wp::float32>(var_107, var_12);
                    wp::array_store(var_24, var_6, var_108);
                }
                if (!var_105) {
                    // elif jnt_type_ == JointType.HINGE:  # hinge                                <L 820>
                    var_110 = (var_9 == var_109);
                    if (var_110) {
                        // res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))         <L 821>
                        var_111 = wp::cross(var_12, var_21);
                        var_112 = wp::vec_t<6, wp::float32>(var_12, var_111);
                        wp::array_store(var_24, var_6, var_112);
                    }
                }
            }
        }
    }
}



extern "C" __global__ void _flex_edges_3e995a68_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_edgeadr,
    wp::array_t<wp::int32> var_flex_edgenum,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::int32> var_flexedge_J_rowadr,
    wp::array_t<wp::int32> var_flexedge_J_colind,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::float32> var_flexedge_J_out,
    wp::array_t<wp::float32> var_flexedge_length_out,
    wp::array_t<wp::float32> var_flexedge_velocity_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::float32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        bool var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        const wp::float32 var_46 = 0.0;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        const wp::float32 var_53 = 0.0;
        wp::float32 var_54;
        const wp::int32 var_55 = 0;
        bool var_56;
        wp::int32* var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::vec_t<3, wp::float32>* var_61;
        wp::int32 var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::range_t var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::vec_t<6, wp::float32>* var_69;
        wp::vec_t<6, wp::float32> var_70;
        wp::vec_t<6, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::float32 var_76;
        wp::int32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 0;
        bool var_83;
        wp::int32* var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::int32* var_87;
        wp::vec_t<3, wp::float32>* var_88;
        wp::int32 var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::range_t var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::vec_t<6, wp::float32>* var_96;
        wp::vec_t<6, wp::float32> var_97;
        wp::vec_t<6, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::float32 var_103;
        wp::int32 var_104;
        wp::float32* var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::int32 var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        const wp::int32 var_113 = 0;
        const wp::int32 var_114 = 0;
        bool var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32* var_119;
        wp::vec_t<3, wp::float32>* var_120;
        wp::int32 var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::range_t var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::vec_t<6, wp::float32>* var_128;
        wp::vec_t<6, wp::float32> var_129;
        wp::vec_t<6, wp::float32> var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::vec_t<3, wp::float32> var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::float32 var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::int32 var_142;
        wp::int32 var_143;
        const wp::int32 var_144 = 0;
        bool var_145;
        wp::int32* var_146;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::int32* var_149;
        wp::vec_t<3, wp::float32>* var_150;
        wp::int32 var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::range_t var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::vec_t<6, wp::float32>* var_158;
        wp::vec_t<6, wp::float32> var_159;
        wp::vec_t<6, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::vec_t<3, wp::float32> var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::float32 var_165;
        wp::int32 var_166;
        wp::int32 var_167;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::vec_t<3, wp::float32> var_170;
        //---------
        // forward
        // def _flex_edges(                                                                       <L 346>
        // worldid, edgeid = wp.tid()                                                             <L 369>
        builtin_tid2d(var_0, var_1);
        // for i in range(nflex):                                                                 <L 370>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = edgeid - flex_edgeadr[i]                                                   <L 371>
            var_4 = wp::address(var_flex_edgeadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_edgenum[i]:                                         <L 372>
            var_9 = (var_5 >= var_8);
            var_7 = var_9;
            if (var_7) {
                var_10 = wp::address(var_flex_edgenum, var_3);
                var_12 = wp::load(var_10);
                var_11 = (var_5 < var_12);
                var_7 = var_7 && var_11;
            }
            if (var_7) {
                // f = i                                                                          <L 373>
                var_13 = wp::copy(var_3);
                // break                                                                          <L 374>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // vbase = flex_vertadr[f]                                                                <L 376>
        var_14 = wp::address(var_flex_vertadr, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // v = flex_edge[edgeid]                                                                  <L 377>
        var_17 = wp::address(var_flex_edge, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // vbase0 = vbase + v[0]                                                                  <L 378>
        var_21 = wp::extract(var_18, var_20);
        var_22 = wp::add(var_15, var_21);
        // vbase1 = vbase + v[1]                                                                  <L 379>
        var_24 = wp::extract(var_18, var_23);
        var_25 = wp::add(var_15, var_24);
        // pos1 = flexvert_xpos_in[worldid, vbase0]                                               <L 381>
        var_26 = wp::address(var_flexvert_xpos_in, var_0, var_22);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // pos2 = flexvert_xpos_in[worldid, vbase1]                                               <L 382>
        var_29 = wp::address(var_flexvert_xpos_in, var_0, var_25);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // vec = pos2 - pos1                                                                      <L 383>
        var_32 = wp::sub(var_30, var_27);
        // edge, edge_length = math.normalize_with_norm(vec)                                      <L 384>
        normalize_with_norm_0(var_32, var_33, var_34);
        // flexedge_length_out[worldid, edgeid] = edge_length                                     <L 385>
        wp::array_store(var_flexedge_length_out, var_0, var_1, var_34);
        // b1 = flex_vertbodyid[vbase0]                                                           <L 387>
        var_35 = wp::address(var_flex_vertbodyid, var_22);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // b2 = flex_vertbodyid[vbase1]                                                           <L 388>
        var_38 = wp::address(var_flex_vertbodyid, var_25);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // if b1 < 0 or b2 < 0:                                                                   <L 391>
        var_43 = (var_36 < var_42);
        var_41 = var_43;
        if (!var_41) {
            var_45 = (var_39 < var_44);
            var_41 = var_41 || var_45;
        }
        if (var_41) {
            // flexedge_velocity_out[worldid, edgeid] = 0.0                                       <L 392>
            wp::array_store(var_flexedge_velocity_out, var_0, var_1, var_46);
            // return                                                                             <L 393>
            continue;
        }
        // dofnum1 = body_dofnum[b1]                                                              <L 395>
        var_47 = wp::address(var_body_dofnum, var_36);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // dofnum2 = body_dofnum[b2]                                                              <L 396>
        var_50 = wp::address(var_body_dofnum, var_39);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // vel = float(0.0)                                                                       <L 399>
        var_54 = wp::float(var_53);
        // if dofnum1 > 0:                                                                        <L 400>
        var_56 = (var_48 > var_55);
        if (var_56) {
            // dofi = body_dofadr[b1]                                                             <L 401>
            var_57 = wp::address(var_body_dofadr, var_36);
            var_59 = wp::load(var_57);
            var_58 = wp::copy(var_59);
            // offset1 = pos1 - wp.vec3(subtree_com_in[worldid, body_rootid[b1]])                 <L 402>
            var_60 = wp::address(var_body_rootid, var_36);
            var_62 = wp::load(var_60);
            var_61 = wp::address(var_subtree_com_in, var_0, var_62);
            var_64 = wp::load(var_61);
            var_63 = wp::vec_t<3, wp::float32>(var_64);
            var_65 = wp::sub(var_27, var_63);
            // for k in range(dofnum1):                                                           <L 403>
            var_66 = wp::range(var_48);
            start_for_3:;
                if (iter_cmp(var_66) == 0) goto end_for_3;
                var_67 = wp::iter_next(var_66);
                // cdof = cdof_in[worldid, dofi + k]                                              <L 404>
                var_68 = wp::add(var_58, var_67);
                var_69 = wp::address(var_cdof_in, var_0, var_68);
                var_71 = wp::load(var_69);
                var_70 = wp::copy(var_71);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 405>
                var_72 = wp::spatial_top(var_70);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 406>
                var_73 = wp::spatial_bottom(var_70);
                // jacp1 = cdof_lin + wp.cross(cdof_ang, offset1)                                 <L 407>
                var_74 = wp::cross(var_72, var_65);
                var_75 = wp::add(var_73, var_74);
                // vel -= wp.dot(jacp1, edge) * qvel_in[worldid, dofi + k]                        <L 408>
                var_76 = wp::dot(var_75, var_33);
                var_77 = wp::add(var_58, var_67);
                var_78 = wp::address(var_qvel_in, var_0, var_77);
                var_80 = wp::load(var_78);
                var_79 = wp::mul(var_76, var_80);
                var_81 = wp::sub(var_54, var_79);
                wp::assign(var_54, var_81);
                goto start_for_3;
            end_for_3:;
        }
        // if dofnum2 > 0:                                                                        <L 409>
        var_83 = (var_51 > var_82);
        if (var_83) {
            // dofj = body_dofadr[b2]                                                             <L 410>
            var_84 = wp::address(var_body_dofadr, var_39);
            var_86 = wp::load(var_84);
            var_85 = wp::copy(var_86);
            // offset2 = pos2 - wp.vec3(subtree_com_in[worldid, body_rootid[b2]])                 <L 411>
            var_87 = wp::address(var_body_rootid, var_39);
            var_89 = wp::load(var_87);
            var_88 = wp::address(var_subtree_com_in, var_0, var_89);
            var_91 = wp::load(var_88);
            var_90 = wp::vec_t<3, wp::float32>(var_91);
            var_92 = wp::sub(var_30, var_90);
            // for k in range(dofnum2):                                                           <L 412>
            var_93 = wp::range(var_51);
            start_for_5:;
                if (iter_cmp(var_93) == 0) goto end_for_5;
                var_94 = wp::iter_next(var_93);
                // cdof = cdof_in[worldid, dofj + k]                                              <L 413>
                var_95 = wp::add(var_85, var_94);
                var_96 = wp::address(var_cdof_in, var_0, var_95);
                var_98 = wp::load(var_96);
                var_97 = wp::copy(var_98);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 414>
                var_99 = wp::spatial_top(var_97);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 415>
                var_100 = wp::spatial_bottom(var_97);
                // jacp2 = cdof_lin + wp.cross(cdof_ang, offset2)                                 <L 416>
                var_101 = wp::cross(var_99, var_92);
                var_102 = wp::add(var_100, var_101);
                // vel += wp.dot(jacp2, edge) * qvel_in[worldid, dofj + k]                        <L 417>
                var_103 = wp::dot(var_102, var_33);
                var_104 = wp::add(var_85, var_94);
                var_105 = wp::address(var_qvel_in, var_0, var_104);
                var_107 = wp::load(var_105);
                var_106 = wp::mul(var_103, var_107);
                var_108 = wp::add(var_54, var_106);
                wp::assign(var_54, var_108);
                wp::assign(var_70, var_97);
                wp::assign(var_72, var_99);
                wp::assign(var_73, var_100);
                goto start_for_5;
            end_for_5:;
        }
        var_109 = wp::where(var_83, var_94, var_67);
        // flexedge_velocity_out[worldid, edgeid] = vel                                           <L 418>
        wp::array_store(var_flexedge_velocity_out, var_0, var_1, var_54);
        // rowadr = flexedge_J_rowadr[edgeid]                                                     <L 420>
        var_110 = wp::address(var_flexedge_J_rowadr, var_1);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // nnz_offset = 0                                                                         <L 421>
        // if dofnum1 > 0:                                                                        <L 424>
        var_115 = (var_48 > var_114);
        if (var_115) {
            // dofi = body_dofadr[b1]                                                             <L 425>
            var_116 = wp::address(var_body_dofadr, var_36);
            var_118 = wp::load(var_116);
            var_117 = wp::copy(var_118);
            // offset1 = pos1 - wp.vec3(subtree_com_in[worldid, body_rootid[b1]])                 <L 426>
            var_119 = wp::address(var_body_rootid, var_36);
            var_121 = wp::load(var_119);
            var_120 = wp::address(var_subtree_com_in, var_0, var_121);
            var_123 = wp::load(var_120);
            var_122 = wp::vec_t<3, wp::float32>(var_123);
            var_124 = wp::sub(var_27, var_122);
            // for k in range(dofnum1):                                                           <L 427>
            var_125 = wp::range(var_48);
            start_for_7:;
                if (iter_cmp(var_125) == 0) goto end_for_7;
                var_126 = wp::iter_next(var_125);
                // cdof = cdof_in[worldid, dofi + k]                                              <L 428>
                var_127 = wp::add(var_117, var_126);
                var_128 = wp::address(var_cdof_in, var_0, var_127);
                var_130 = wp::load(var_128);
                var_129 = wp::copy(var_130);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 429>
                var_131 = wp::spatial_top(var_129);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 430>
                var_132 = wp::spatial_bottom(var_129);
                // jacp1 = cdof_lin + wp.cross(cdof_ang, offset1)                                 <L 431>
                var_133 = wp::cross(var_131, var_124);
                var_134 = wp::add(var_132, var_133);
                // flexedge_J_out[worldid, rowadr + nnz_offset + k] = wp.dot(-jacp1, edge)        <L 432>
                var_135 = wp::neg(var_134);
                var_136 = wp::dot(var_135, var_33);
                var_137 = wp::add(var_111, var_113);
                var_138 = wp::add(var_137, var_126);
                wp::array_store(var_flexedge_J_out, var_0, var_138, var_136);
                wp::assign(var_70, var_129);
                wp::assign(var_72, var_131);
                wp::assign(var_73, var_132);
                wp::assign(var_75, var_134);
                goto start_for_7;
            end_for_7:;
            // nnz_offset += dofnum1                                                              <L 433>
            var_139 = wp::add(var_113, var_48);
        }
        var_140 = wp::where(var_115, var_117, var_58);
        var_141 = wp::where(var_115, var_124, var_65);
        var_142 = wp::where(var_115, var_126, var_109);
        var_143 = wp::where(var_115, var_139, var_113);
        // if dofnum2 > 0:                                                                        <L 436>
        var_145 = (var_51 > var_144);
        if (var_145) {
            // dofj = body_dofadr[b2]                                                             <L 437>
            var_146 = wp::address(var_body_dofadr, var_39);
            var_148 = wp::load(var_146);
            var_147 = wp::copy(var_148);
            // offset2 = pos2 - wp.vec3(subtree_com_in[worldid, body_rootid[b2]])                 <L 438>
            var_149 = wp::address(var_body_rootid, var_39);
            var_151 = wp::load(var_149);
            var_150 = wp::address(var_subtree_com_in, var_0, var_151);
            var_153 = wp::load(var_150);
            var_152 = wp::vec_t<3, wp::float32>(var_153);
            var_154 = wp::sub(var_30, var_152);
            // for k in range(dofnum2):                                                           <L 439>
            var_155 = wp::range(var_51);
            start_for_9:;
                if (iter_cmp(var_155) == 0) goto end_for_9;
                var_156 = wp::iter_next(var_155);
                // cdof = cdof_in[worldid, dofj + k]                                              <L 440>
                var_157 = wp::add(var_147, var_156);
                var_158 = wp::address(var_cdof_in, var_0, var_157);
                var_160 = wp::load(var_158);
                var_159 = wp::copy(var_160);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 441>
                var_161 = wp::spatial_top(var_159);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 442>
                var_162 = wp::spatial_bottom(var_159);
                // jacp2 = cdof_lin + wp.cross(cdof_ang, offset2)                                 <L 443>
                var_163 = wp::cross(var_161, var_154);
                var_164 = wp::add(var_162, var_163);
                // flexedge_J_out[worldid, rowadr + nnz_offset + k] = wp.dot(jacp2, edge)         <L 444>
                var_165 = wp::dot(var_164, var_33);
                var_166 = wp::add(var_111, var_143);
                var_167 = wp::add(var_166, var_156);
                wp::array_store(var_flexedge_J_out, var_0, var_167, var_165);
                wp::assign(var_70, var_159);
                wp::assign(var_72, var_161);
                wp::assign(var_73, var_162);
                wp::assign(var_102, var_164);
                goto start_for_9;
            end_for_9:;
        }
        var_168 = wp::where(var_145, var_156, var_142);
        var_169 = wp::where(var_145, var_147, var_85);
        var_170 = wp::where(var_145, var_154, var_92);
    }
}



extern "C" __global__ void _qfrc_bias_4085b6de_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::float32> var_qfrc_bias_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::float32 var_7;
        wp::vec_t<6, wp::float32> var_8;
        wp::vec_t<6, wp::float32> var_9;
        //---------
        // forward
        // def _qfrc_bias(                                                                        <L 1485>
        // worldid, dofid = wp.tid()                                                              <L 1494>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 1495>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc_bias_out[worldid, dofid] = wp.dot(cdof_in[worldid, dofid], cfrc_int_in[worldid, bodyid])       <L 1496>
        var_5 = wp::address(var_cdof_in, var_0, var_1);
        var_6 = wp::address(var_cfrc_int_in, var_0, var_3);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::dot(var_8, var_9);
        wp::array_store(var_qfrc_bias_out, var_0, var_1, var_7);
    }
}



extern "C" __global__ void _transmission_4db1a205_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_actuator_cranklength,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::int32> var_moment_nnz,
    wp::array_t<wp::float32> var_actuator_length_out,
    wp::array_t<wp::int32> var_moment_rownnz_out,
    wp::array_t<wp::int32> var_moment_rowadr_out,
    wp::array_t<wp::int32> var_moment_colind_out,
    wp::array_t<wp::float32> var_actuator_moment_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::array_t<wp::float32> var_18;
        wp::vec_t<2, wp::int32>* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::vec_t<2, wp::int32> var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 6;
        const wp::int32 var_35 = 6;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        const wp::int32 var_47 = 2;
        wp::int32 var_48;
        const wp::int32 var_49 = 3;
        wp::int32 var_50;
        const wp::int32 var_51 = 3;
        wp::int32 var_52;
        const wp::int32 var_53 = 4;
        wp::int32 var_54;
        const wp::int32 var_55 = 4;
        wp::int32 var_56;
        const wp::int32 var_57 = 5;
        wp::int32 var_58;
        const wp::int32 var_59 = 5;
        wp::int32 var_60;
        const wp::float32 var_61 = 0.0;
        const wp::int32 var_62 = 1;
        bool var_63;
        const wp::int32 var_64 = 3;
        wp::int32 var_65;
        wp::float32* var_66;
        const wp::int32 var_67 = 4;
        wp::int32 var_68;
        wp::float32* var_69;
        const wp::int32 var_70 = 5;
        wp::int32 var_71;
        wp::float32* var_72;
        const wp::int32 var_73 = 6;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::quat_t<wp::float32> var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::quat_t<wp::float32> var_81;
        wp::quat_t<wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        const wp::int32 var_85 = 0;
        wp::float32 var_86;
        const wp::int32 var_87 = 0;
        wp::int32 var_88;
        const wp::int32 var_89 = 1;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::int32 var_92;
        const wp::int32 var_93 = 2;
        wp::float32 var_94;
        const wp::int32 var_95 = 2;
        wp::int32 var_96;
        const wp::int32 var_97 = 0;
        wp::float32 var_98;
        const wp::int32 var_99 = 3;
        wp::int32 var_100;
        const wp::int32 var_101 = 1;
        wp::float32 var_102;
        const wp::int32 var_103 = 4;
        wp::int32 var_104;
        const wp::int32 var_105 = 2;
        wp::float32 var_106;
        const wp::int32 var_107 = 5;
        wp::int32 var_108;
        const wp::int32 var_109 = 0;
        wp::float32 var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        const wp::int32 var_113 = 1;
        wp::float32 var_114;
        const wp::int32 var_115 = 1;
        wp::int32 var_116;
        const wp::int32 var_117 = 2;
        wp::float32 var_118;
        const wp::int32 var_119 = 2;
        wp::int32 var_120;
        const wp::int32 var_121 = 3;
        wp::float32 var_122;
        const wp::int32 var_123 = 3;
        wp::int32 var_124;
        const wp::int32 var_125 = 4;
        wp::float32 var_126;
        const wp::int32 var_127 = 4;
        wp::int32 var_128;
        const wp::int32 var_129 = 5;
        wp::float32 var_130;
        const wp::int32 var_131 = 5;
        wp::int32 var_132;
        const wp::int32 var_133 = 1;
        bool var_134;
        const wp::int32 var_135 = 0;
        wp::int32 var_136;
        wp::float32* var_137;
        const wp::int32 var_138 = 1;
        wp::int32 var_139;
        wp::float32* var_140;
        const wp::int32 var_141 = 2;
        wp::int32 var_142;
        wp::float32* var_143;
        const wp::int32 var_144 = 3;
        wp::int32 var_145;
        wp::float32* var_146;
        wp::quat_t<wp::float32> var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::quat_t<wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        const wp::int32 var_155 = 1;
        bool var_156;
        wp::quat_t<wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 3;
        wp::int32 var_162;
        const wp::int32 var_163 = 0;
        wp::int32 var_164;
        wp::int32 var_165;
        wp::float32 var_166;
        const wp::int32 var_167 = 1;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 2;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::float32 var_174;
        bool var_175;
        const wp::int32 var_176 = 2;
        bool var_177;
        const wp::int32 var_178 = 3;
        bool var_179;
        wp::float32* var_180;
        const wp::int32 var_181 = 0;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        const wp::int32 var_185 = 1;
        wp::int32 var_186;
        const wp::int32 var_187 = 0;
        wp::float32 var_188;
        const wp::str var_189 = "unrecognized joint type";
        wp::int32 var_190;
        wp::int32 var_191;
        wp::int32 var_192;
        wp::quat_t<wp::float32> var_193;
        wp::vec_t<3, wp::float32> var_194;
        const wp::int32 var_195 = 2;
        bool var_196;
        wp::vec_t<2, wp::int32>* var_197;
        wp::vec_t<2, wp::int32> var_198;
        wp::vec_t<2, wp::int32> var_199;
        const wp::int32 var_200 = 0;
        wp::int32 var_201;
        const wp::int32 var_202 = 1;
        wp::int32 var_203;
        const wp::int32 var_204 = 0;
        wp::float32 var_205;
        wp::shape_t* var_206;
        const wp::int32 var_207 = 0;
        wp::int32 var_208;
        wp::shape_t var_209;
        wp::int32 var_210;
        wp::float32* var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::mat_t<3, 3, wp::float32>* var_214;
        wp::mat_t<3, 3, wp::float32> var_215;
        wp::mat_t<3, 3, wp::float32> var_216;
        const wp::int32 var_217 = 0;
        const wp::int32 var_218 = 2;
        wp::float32 var_219;
        const wp::int32 var_220 = 1;
        const wp::int32 var_221 = 2;
        wp::float32 var_222;
        const wp::int32 var_223 = 2;
        const wp::int32 var_224 = 2;
        wp::float32 var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32>* var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32>* var_230;
        wp::vec_t<3, wp::float32> var_231;
        wp::vec_t<3, wp::float32> var_232;
        wp::vec_t<3, wp::float32> var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::float32 var_238;
        wp::float32 var_239;
        const wp::int32 var_240 = 1;
        const wp::float32 var_241 = 0.0;
        bool var_242;
        const wp::int32 var_243 = 0;
        const wp::float32 var_244 = 0.0;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::int32 var_248;
        wp::float32 var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        const wp::int32 var_252 = 1;
        bool var_253;
        const wp::float32 var_254 = 1.0;
        wp::float32 var_255;
        wp::float32 var_256;
        wp::vec_t<3, wp::float32> var_257;
        wp::vec_t<3, wp::float32> var_258;
        wp::vec_t<3, wp::float32> var_259;
        wp::vec_t<3, wp::float32> var_260;
        wp::vec_t<3, wp::float32> var_261;
        wp::vec_t<3, wp::float32> var_262;
        wp::vec_t<3, wp::float32> var_263;
        wp::vec_t<3, wp::float32> var_264;
        wp::int32* var_265;
        wp::int32* var_266;
        wp::int32 var_267;
        wp::int32 var_268;
        wp::int32 var_269;
        wp::int32* var_270;
        wp::int32* var_271;
        wp::int32 var_272;
        wp::int32 var_273;
        wp::int32 var_274;
        const wp::int32 var_275 = -1;
        wp::int32 var_276;
        const wp::int32 var_277 = -1;
        wp::int32 var_278;
        const wp::int32 var_279 = 0;
        bool var_280;
        wp::int32* var_281;
        wp::int32* var_282;
        wp::int32 var_283;
        wp::int32 var_284;
        wp::int32 var_285;
        const wp::int32 var_286 = 1;
        wp::int32 var_287;
        wp::int32 var_288;
        const wp::int32 var_289 = 0;
        bool var_290;
        wp::int32* var_291;
        wp::int32* var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        const wp::int32 var_296 = 1;
        wp::int32 var_297;
        wp::int32 var_298;
        wp::int32 var_299;
        wp::int32 var_300;
        const wp::int32 var_301 = 0;
        wp::int32 var_302;
        bool var_303;
        const wp::int32 var_304 = 0;
        bool var_305;
        const wp::int32 var_306 = 0;
        bool var_307;
        wp::int32 var_308;
        const wp::int32 var_309 = 1;
        wp::int32 var_310;
        bool var_311;
        wp::int32* var_312;
        wp::int32 var_313;
        wp::int32 var_314;
        wp::int32 var_315;
        bool var_316;
        wp::int32* var_317;
        wp::int32 var_318;
        wp::int32 var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        wp::int32 var_323;
        const wp::int32 var_324 = 1;
        wp::int32 var_325;
        bool var_326;
        const wp::int32 var_327 = 0;
        bool var_328;
        const wp::int32 var_329 = 0;
        bool var_330;
        wp::int32 var_331;
        wp::int32* var_332;
        wp::vec_t<3, wp::float32> var_333;
        wp::vec_t<3, wp::float32> var_334;
        wp::int32 var_335;
        wp::vec_t<3, wp::float32> var_336;
        wp::vec_t<3, wp::float32> var_337;
        wp::int32* var_338;
        wp::vec_t<3, wp::float32> var_339;
        wp::vec_t<3, wp::float32> var_340;
        wp::int32 var_341;
        wp::vec_t<3, wp::float32> var_342;
        wp::float32 var_343;
        wp::float32 var_344;
        wp::float32 var_345;
        wp::int32 var_346;
        wp::float32 var_347;
        const wp::int32 var_348 = 1;
        wp::int32 var_349;
        bool var_350;
        wp::int32* var_351;
        wp::int32 var_352;
        wp::int32 var_353;
        wp::int32 var_354;
        bool var_355;
        wp::int32* var_356;
        wp::int32 var_357;
        wp::int32 var_358;
        wp::int32 var_359;
        const wp::int32 var_360 = 3;
        bool var_361;
        wp::vec_t<2, wp::int32>* var_362;
        const wp::int32 var_363 = 0;
        wp::int32 var_364;
        wp::vec_t<2, wp::int32> var_365;
        const wp::int32 var_366 = 0;
        wp::float32 var_367;
        wp::float32* var_368;
        wp::float32 var_369;
        wp::float32 var_370;
        wp::int32* var_371;
        wp::int32 var_372;
        wp::int32 var_373;
        wp::int32* var_374;
        wp::int32 var_375;
        wp::int32 var_376;
        wp::int32 var_377;
        wp::range_t var_378;
        wp::int32 var_379;
        wp::int32 var_380;
        wp::int32 var_381;
        wp::int32* var_382;
        wp::int32 var_383;
        wp::float32* var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        const wp::int32 var_387 = 5;
        bool var_388;
        const wp::float32 var_389 = 0.0;
        wp::int32 var_390;
        wp::range_t var_391;
        wp::int32 var_392;
        wp::int32 var_393;
        const wp::float32 var_394 = 0.0;
        const wp::int32 var_395 = 4;
        bool var_396;
        wp::vec_t<2, wp::int32>* var_397;
        wp::vec_t<2, wp::int32> var_398;
        wp::vec_t<2, wp::int32> var_399;
        const wp::int32 var_400 = 0;
        wp::int32 var_401;
        const wp::int32 var_402 = 1;
        wp::int32 var_403;
        wp::vec_t<6, wp::float32>* var_404;
        wp::vec_t<6, wp::float32> var_405;
        wp::vec_t<6, wp::float32> var_406;
        wp::shape_t* var_407;
        const wp::int32 var_408 = 0;
        wp::int32 var_409;
        wp::shape_t var_410;
        wp::int32 var_411;
        wp::vec_t<3, wp::float32> var_412;
        wp::vec_t<3, wp::float32> var_413;
        const wp::int32 var_414 = -1;
        bool var_415;
        wp::mat_t<3, 3, wp::float32>* var_416;
        wp::mat_t<3, 3, wp::float32> var_417;
        wp::mat_t<3, 3, wp::float32> var_418;
        wp::vec_t<3, wp::float32> var_419;
        wp::vec_t<3, wp::float32> var_420;
        wp::int32* var_421;
        wp::int32* var_422;
        wp::int32 var_423;
        wp::int32 var_424;
        wp::int32 var_425;
        const wp::int32 var_426 = -1;
        wp::int32 var_427;
        const wp::int32 var_428 = 0;
        bool var_429;
        wp::int32* var_430;
        wp::int32* var_431;
        wp::int32 var_432;
        wp::int32 var_433;
        wp::int32 var_434;
        const wp::int32 var_435 = 1;
        wp::int32 var_436;
        wp::int32 var_437;
        wp::int32 var_438;
        const wp::int32 var_439 = 0;
        wp::int32 var_440;
        const wp::int32 var_441 = 0;
        bool var_442;
        const wp::int32 var_443 = 1;
        wp::int32 var_444;
        wp::int32* var_445;
        wp::int32 var_446;
        wp::int32 var_447;
        wp::int32 var_448;
        const wp::float32 var_449 = 0.0;
        wp::int32 var_450;
        const wp::int32 var_451 = 1;
        wp::int32 var_452;
        const wp::int32 var_453 = 0;
        bool var_454;
        wp::vec_t<3, wp::float32>* var_455;
        wp::int32* var_456;
        wp::vec_t<3, wp::float32> var_457;
        wp::vec_t<3, wp::float32> var_458;
        wp::vec_t<3, wp::float32> var_459;
        wp::int32 var_460;
        wp::float32 var_461;
        wp::float32 var_462;
        wp::float32 var_463;
        wp::int32 var_464;
        const wp::int32 var_465 = 1;
        wp::int32 var_466;
        wp::int32* var_467;
        wp::int32 var_468;
        wp::int32 var_469;
        wp::int32* var_470;
        wp::int32 var_471;
        wp::int32 var_472;
        wp::int32* var_473;
        wp::int32 var_474;
        wp::int32 var_475;
        wp::int32* var_476;
        wp::int32 var_477;
        wp::int32 var_478;
        wp::int32* var_479;
        wp::int32 var_480;
        wp::int32 var_481;
        wp::int32* var_482;
        wp::int32* var_483;
        wp::int32 var_484;
        wp::int32 var_485;
        wp::int32 var_486;
        const wp::int32 var_487 = 1;
        wp::int32 var_488;
        wp::int32* var_489;
        wp::int32* var_490;
        wp::int32 var_491;
        wp::int32 var_492;
        wp::int32 var_493;
        const wp::int32 var_494 = 1;
        wp::int32 var_495;
        const wp::int32 var_496 = -1;
        bool var_497;
        const wp::int32 var_498 = 0;
        bool var_499;
        const wp::int32 var_500 = 0;
        bool var_501;
        bool var_502;
        bool var_503;
        wp::int32* var_504;
        wp::int32 var_505;
        wp::int32 var_506;
        wp::int32* var_507;
        wp::int32 var_508;
        wp::int32 var_509;
        wp::int32 var_510;
        wp::int32 var_511;
        bool var_512;
        const wp::int32 var_513 = -1;
        bool var_514;
        const wp::int32 var_515 = -1;
        bool var_516;
        wp::int32 var_517;
        wp::int32 var_518;
        bool var_519;
        wp::int32 var_520;
        wp::int32 var_521;
        wp::int32 var_522;
        bool var_523;
        const wp::int32 var_524 = 0;
        wp::float32 var_525;
        const wp::float32 var_526 = 0.0;
        bool var_527;
        const wp::int32 var_528 = 1;
        wp::float32 var_529;
        const wp::float32 var_530 = 0.0;
        bool var_531;
        const wp::int32 var_532 = 2;
        wp::float32 var_533;
        const wp::float32 var_534 = 0.0;
        bool var_535;
        bool var_536;
        bool var_537;
        const wp::int32 var_538 = 3;
        wp::float32 var_539;
        const wp::float32 var_540 = 0.0;
        bool var_541;
        const wp::int32 var_542 = 4;
        wp::float32 var_543;
        const wp::float32 var_544 = 0.0;
        bool var_545;
        const wp::int32 var_546 = 5;
        wp::float32 var_547;
        const wp::float32 var_548 = 0.0;
        bool var_549;
        bool var_550;
        wp::vec_t<3, wp::float32>* var_551;
        wp::vec_t<3, wp::float32> var_552;
        wp::vec_t<3, wp::float32> var_553;
        wp::vec_t<3, wp::float32>* var_554;
        wp::vec_t<3, wp::float32> var_555;
        wp::vec_t<3, wp::float32> var_556;
        wp::mat_t<3, 3, wp::float32>* var_557;
        wp::mat_t<3, 3, wp::float32> var_558;
        wp::mat_t<3, 3, wp::float32> var_559;
        const wp::float32 var_560 = 0.0;
        wp::float32 var_561;
        wp::mat_t<3, 3, wp::float32> var_562;
        wp::vec_t<3, wp::float32> var_563;
        wp::vec_t<3, wp::float32> var_564;
        wp::float32 var_565;
        wp::float32 var_566;
        wp::vec_t<3, wp::float32> var_567;
        wp::float32 var_568;
        wp::quat_t<wp::float32>* var_569;
        wp::quat_t<wp::float32>* var_570;
        wp::quat_t<wp::float32> var_571;
        wp::quat_t<wp::float32> var_572;
        wp::quat_t<wp::float32> var_573;
        wp::quat_t<wp::float32>* var_574;
        wp::quat_t<wp::float32>* var_575;
        wp::quat_t<wp::float32> var_576;
        wp::quat_t<wp::float32> var_577;
        wp::quat_t<wp::float32> var_578;
        wp::vec_t<3, wp::float32> var_579;
        wp::float32 var_580;
        wp::float32 var_581;
        wp::vec_t<3, wp::float32> var_582;
        wp::float32 var_583;
        wp::vec_t<3, wp::float32> var_584;
        const wp::int32 var_585 = -1;
        wp::int32 var_586;
        const wp::int32 var_587 = -1;
        wp::int32 var_588;
        const wp::int32 var_589 = 0;
        bool var_590;
        wp::int32* var_591;
        wp::int32* var_592;
        wp::int32 var_593;
        wp::int32 var_594;
        wp::int32 var_595;
        const wp::int32 var_596 = 1;
        wp::int32 var_597;
        wp::int32 var_598;
        const wp::int32 var_599 = 0;
        bool var_600;
        wp::int32* var_601;
        wp::int32* var_602;
        wp::int32 var_603;
        wp::int32 var_604;
        wp::int32 var_605;
        const wp::int32 var_606 = 1;
        wp::int32 var_607;
        wp::int32 var_608;
        wp::int32 var_609;
        wp::int32 var_610;
        const wp::int32 var_611 = 0;
        wp::int32 var_612;
        bool var_613;
        const wp::int32 var_614 = 0;
        bool var_615;
        const wp::int32 var_616 = 0;
        bool var_617;
        wp::int32 var_618;
        bool var_619;
        bool var_620;
        bool var_621;
        const wp::int32 var_622 = 1;
        wp::int32 var_623;
        bool var_624;
        wp::int32* var_625;
        wp::int32 var_626;
        wp::int32 var_627;
        wp::int32 var_628;
        bool var_629;
        wp::int32* var_630;
        wp::int32 var_631;
        wp::int32 var_632;
        wp::int32 var_633;
        wp::int32 var_634;
        wp::int32 var_635;
        wp::int32 var_636;
        const wp::int32 var_637 = 1;
        wp::int32 var_638;
        bool var_639;
        const wp::int32 var_640 = 0;
        bool var_641;
        const wp::int32 var_642 = 0;
        bool var_643;
        wp::int32 var_644;
        bool var_645;
        bool var_646;
        bool var_647;
        wp::int32 var_648;
        wp::int32* var_649;
        wp::vec_t<3, wp::float32> var_650;
        wp::vec_t<3, wp::float32> var_651;
        wp::int32 var_652;
        wp::int32* var_653;
        wp::vec_t<3, wp::float32> var_654;
        wp::vec_t<3, wp::float32> var_655;
        wp::int32 var_656;
        const wp::float32 var_657 = 0.0;
        wp::float32 var_658;
        wp::vec_t<3, wp::float32> var_659;
        wp::float32 var_660;
        wp::float32 var_661;
        wp::float32 var_662;
        wp::vec_t<3, wp::float32> var_663;
        wp::float32 var_664;
        wp::float32 var_665;
        wp::float32 var_666;
        wp::int32 var_667;
        const wp::int32 var_668 = 1;
        wp::int32 var_669;
        bool var_670;
        wp::int32* var_671;
        wp::int32 var_672;
        wp::int32 var_673;
        wp::int32 var_674;
        bool var_675;
        wp::int32* var_676;
        wp::int32 var_677;
        wp::int32 var_678;
        wp::int32 var_679;
        wp::vec_t<3, wp::float32> var_680;
        wp::vec_t<3, wp::float32> var_681;
        wp::int32 var_682;
        wp::int32 var_683;
        wp::int32 var_684;
        wp::int32 var_685;
        wp::int32 var_686;
        wp::vec_t<3, wp::float32> var_687;
        wp::vec_t<3, wp::float32> var_688;
        wp::float32 var_689;
        wp::int32 var_690;
        const wp::str var_691 = "unhandled transmission type %d\n";
        wp::vec_t<6, wp::float32> var_692;
        wp::vec_t<6, wp::float32> var_693;
        wp::int32 var_694;
        wp::int32 var_695;
        wp::vec_t<6, wp::float32> var_696;
        wp::vec_t<6, wp::float32> var_697;
        wp::vec_t<2, wp::int32> var_698;
        wp::float32 var_699;
        wp::mat_t<3, 3, wp::float32> var_700;
        wp::vec_t<3, wp::float32> var_701;
        wp::float32 var_702;
        wp::int32 var_703;
        wp::int32 var_704;
        wp::int32 var_705;
        wp::int32 var_706;
        wp::int32 var_707;
        wp::int32 var_708;
        wp::int32 var_709;
        wp::int32 var_710;
        wp::int32 var_711;
        wp::vec_t<3, wp::float32> var_712;
        wp::vec_t<3, wp::float32> var_713;
        wp::float32 var_714;
        wp::int32 var_715;
        wp::vec_t<6, wp::float32> var_716;
        wp::int32 var_717;
        wp::quat_t<wp::float32> var_718;
        wp::int32 var_719;
        wp::int32 var_720;
        //---------
        // forward
        // def _transmission(                                                                     <L 2286>
        // worldid, actid = wp.tid()                                                              <L 2327>
        builtin_tid2d(var_0, var_1);
        // trntype = actuator_trntype[actid]                                                      <L 2328>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // actuator_gear_id = worldid % actuator_gear.shape[0]                                    <L 2329>
        var_5 = &(var_actuator_gear.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        // gear = actuator_gear[actuator_gear_id, actid]                                          <L 2330>
        var_10 = wp::address(var_actuator_gear, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 2331>
        var_15 = (var_3 == var_14);
        var_13 = var_15;
        if (!var_13) {
            var_17 = (var_3 == var_16);
            var_13 = var_13 || var_17;
        }
        if (var_13) {
            // qpos = qpos_in[worldid]                                                            <L 2332>
            var_18 = wp::view(var_qpos_in, var_0);
            // jntid = actuator_trnid[actid][0]                                                   <L 2333>
            var_19 = wp::address(var_actuator_trnid, var_1);
            var_22 = wp::load(var_19);
            var_21 = wp::extract(var_22, var_20);
            // jnt_typ = jnt_type[jntid]                                                          <L 2334>
            var_23 = wp::address(var_jnt_type, var_21);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // qadr = jnt_qposadr[jntid]                                                          <L 2335>
            var_26 = wp::address(var_jnt_qposadr, var_21);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // vadr = jnt_dofadr[jntid]                                                           <L 2336>
            var_29 = wp::address(var_jnt_dofadr, var_21);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // if jnt_typ == JointType.FREE:                                                      <L 2337>
            var_33 = (var_24 == var_32);
            if (var_33) {
                // moment_rownnz_out[worldid, actid] = 6                                          <L 2338>
                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_34);
                // rowadr = wp.atomic_add(moment_nnz, worldid, 6)                                 <L 2339>
                var_36 = wp::atomic_add(var_moment_nnz, var_0, var_35);
                // moment_rowadr_out[worldid, actid] = rowadr                                     <L 2340>
                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_36);
                // moment_colind_out[worldid, rowadr + 0] = vadr + 0                              <L 2341>
                var_38 = wp::add(var_30, var_37);
                var_40 = wp::add(var_36, var_39);
                wp::array_store(var_moment_colind_out, var_0, var_40, var_38);
                // moment_colind_out[worldid, rowadr + 1] = vadr + 1                              <L 2342>
                var_42 = wp::add(var_30, var_41);
                var_44 = wp::add(var_36, var_43);
                wp::array_store(var_moment_colind_out, var_0, var_44, var_42);
                // moment_colind_out[worldid, rowadr + 2] = vadr + 2                              <L 2343>
                var_46 = wp::add(var_30, var_45);
                var_48 = wp::add(var_36, var_47);
                wp::array_store(var_moment_colind_out, var_0, var_48, var_46);
                // moment_colind_out[worldid, rowadr + 3] = vadr + 3                              <L 2344>
                var_50 = wp::add(var_30, var_49);
                var_52 = wp::add(var_36, var_51);
                wp::array_store(var_moment_colind_out, var_0, var_52, var_50);
                // moment_colind_out[worldid, rowadr + 4] = vadr + 4                              <L 2345>
                var_54 = wp::add(var_30, var_53);
                var_56 = wp::add(var_36, var_55);
                wp::array_store(var_moment_colind_out, var_0, var_56, var_54);
                // moment_colind_out[worldid, rowadr + 5] = vadr + 5                              <L 2346>
                var_58 = wp::add(var_30, var_57);
                var_60 = wp::add(var_36, var_59);
                wp::array_store(var_moment_colind_out, var_0, var_60, var_58);
                // actuator_length_out[worldid, actid] = 0.0                                      <L 2347>
                wp::array_store(var_actuator_length_out, var_0, var_1, var_61);
                // if trntype == TrnType.JOINTINPARENT:                                           <L 2348>
                var_63 = (var_3 == var_62);
                if (var_63) {
                    // quat = wp.normalize(wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6]))       <L 2349>
                    var_65 = wp::add(var_27, var_64);
                    var_66 = wp::address(var_18, var_65);
                    var_68 = wp::add(var_27, var_67);
                    var_69 = wp::address(var_18, var_68);
                    var_71 = wp::add(var_27, var_70);
                    var_72 = wp::address(var_18, var_71);
                    var_74 = wp::add(var_27, var_73);
                    var_75 = wp::address(var_18, var_74);
                    var_77 = wp::load(var_66);
                    var_78 = wp::load(var_69);
                    var_79 = wp::load(var_72);
                    var_80 = wp::load(var_75);
                    var_76 = wp::quat_t<wp::float32>(var_77, var_78, var_79, var_80);
                    var_81 = wp::normalize(var_76);
                    // quat_neg = math.quat_inv(quat)                                             <L 2350>
                    var_82 = quat_inv_0(var_81);
                    // gearaxis = math.rot_vec_quat(wp.spatial_bottom(gear), quat_neg)            <L 2351>
                    var_83 = wp::spatial_bottom(var_11);
                    var_84 = rot_vec_quat_0(var_83, var_82);
                    // actuator_moment_out[worldid, rowadr + 0] = gear[0]                         <L 2352>
                    var_86 = wp::extract(var_11, var_85);
                    var_88 = wp::add(var_36, var_87);
                    wp::array_store(var_actuator_moment_out, var_0, var_88, var_86);
                    // actuator_moment_out[worldid, rowadr + 1] = gear[1]                         <L 2353>
                    var_90 = wp::extract(var_11, var_89);
                    var_92 = wp::add(var_36, var_91);
                    wp::array_store(var_actuator_moment_out, var_0, var_92, var_90);
                    // actuator_moment_out[worldid, rowadr + 2] = gear[2]                         <L 2354>
                    var_94 = wp::extract(var_11, var_93);
                    var_96 = wp::add(var_36, var_95);
                    wp::array_store(var_actuator_moment_out, var_0, var_96, var_94);
                    // actuator_moment_out[worldid, rowadr + 3] = gearaxis[0]                     <L 2355>
                    var_98 = wp::extract(var_84, var_97);
                    var_100 = wp::add(var_36, var_99);
                    wp::array_store(var_actuator_moment_out, var_0, var_100, var_98);
                    // actuator_moment_out[worldid, rowadr + 4] = gearaxis[1]                     <L 2356>
                    var_102 = wp::extract(var_84, var_101);
                    var_104 = wp::add(var_36, var_103);
                    wp::array_store(var_actuator_moment_out, var_0, var_104, var_102);
                    // actuator_moment_out[worldid, rowadr + 5] = gearaxis[2]                     <L 2357>
                    var_106 = wp::extract(var_84, var_105);
                    var_108 = wp::add(var_36, var_107);
                    wp::array_store(var_actuator_moment_out, var_0, var_108, var_106);
                }
                if (!var_63) {
                    // actuator_moment_out[worldid, rowadr + 0] = gear[0]                         <L 2359>
                    var_110 = wp::extract(var_11, var_109);
                    var_112 = wp::add(var_36, var_111);
                    wp::array_store(var_actuator_moment_out, var_0, var_112, var_110);
                    // actuator_moment_out[worldid, rowadr + 1] = gear[1]                         <L 2360>
                    var_114 = wp::extract(var_11, var_113);
                    var_116 = wp::add(var_36, var_115);
                    wp::array_store(var_actuator_moment_out, var_0, var_116, var_114);
                    // actuator_moment_out[worldid, rowadr + 2] = gear[2]                         <L 2361>
                    var_118 = wp::extract(var_11, var_117);
                    var_120 = wp::add(var_36, var_119);
                    wp::array_store(var_actuator_moment_out, var_0, var_120, var_118);
                    // actuator_moment_out[worldid, rowadr + 3] = gear[3]                         <L 2362>
                    var_122 = wp::extract(var_11, var_121);
                    var_124 = wp::add(var_36, var_123);
                    wp::array_store(var_actuator_moment_out, var_0, var_124, var_122);
                    // actuator_moment_out[worldid, rowadr + 4] = gear[4]                         <L 2363>
                    var_126 = wp::extract(var_11, var_125);
                    var_128 = wp::add(var_36, var_127);
                    wp::array_store(var_actuator_moment_out, var_0, var_128, var_126);
                    // actuator_moment_out[worldid, rowadr + 5] = gear[5]                         <L 2364>
                    var_130 = wp::extract(var_11, var_129);
                    var_132 = wp::add(var_36, var_131);
                    wp::array_store(var_actuator_moment_out, var_0, var_132, var_130);
                }
            }
            if (!var_33) {
                // elif jnt_typ == JointType.BALL:                                                <L 2365>
                var_134 = (var_24 == var_133);
                if (var_134) {
                    // q = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])       <L 2366>
                    var_136 = wp::add(var_27, var_135);
                    var_137 = wp::address(var_18, var_136);
                    var_139 = wp::add(var_27, var_138);
                    var_140 = wp::address(var_18, var_139);
                    var_142 = wp::add(var_27, var_141);
                    var_143 = wp::address(var_18, var_142);
                    var_145 = wp::add(var_27, var_144);
                    var_146 = wp::address(var_18, var_145);
                    var_148 = wp::load(var_137);
                    var_149 = wp::load(var_140);
                    var_150 = wp::load(var_143);
                    var_151 = wp::load(var_146);
                    var_147 = wp::quat_t<wp::float32>(var_148, var_149, var_150, var_151);
                    // q = wp.normalize(q)                                                        <L 2367>
                    var_152 = wp::normalize(var_147);
                    // axis_angle = math.quat_to_vel(q)                                           <L 2368>
                    var_153 = quat_to_vel_0(var_152);
                    // gearaxis = wp.spatial_top(gear)  # [:3]                                    <L 2369>
                    var_154 = wp::spatial_top(var_11);
                    // if trntype == TrnType.JOINTINPARENT:                                       <L 2370>
                    var_156 = (var_3 == var_155);
                    if (var_156) {
                        // quat_neg = math.quat_inv(q)                                            <L 2371>
                        var_157 = quat_inv_0(var_152);
                        // gearaxis = math.rot_vec_quat(gearaxis, quat_neg)                       <L 2372>
                        var_158 = rot_vec_quat_0(var_154, var_157);
                    }
                    var_159 = wp::where(var_156, var_158, var_154);
                    // actuator_length_out[worldid, actid] = wp.dot(axis_angle, gearaxis)         <L 2373>
                    var_160 = wp::dot(var_153, var_159);
                    wp::array_store(var_actuator_length_out, var_0, var_1, var_160);
                    // nnz = 3                                                                    <L 2375>
                    // moment_rownnz_out[worldid, actid] = nnz                                    <L 2376>
                    wp::array_store(var_moment_rownnz_out, var_0, var_1, var_161);
                    // rowadr = wp.atomic_add(moment_nnz, worldid, nnz)                           <L 2377>
                    var_162 = wp::atomic_add(var_moment_nnz, var_0, var_161);
                    // moment_rowadr_out[worldid, actid] = rowadr                                 <L 2378>
                    wp::array_store(var_moment_rowadr_out, var_0, var_1, var_162);
                    // for i in range(3):                                                         <L 2380>
                    // sparseid = rowadr + i                                                      <L 2381>
                    var_164 = wp::add(var_162, var_163);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2382>
                    var_165 = wp::add(var_30, var_163);
                    wp::array_store(var_moment_colind_out, var_0, var_164, var_165);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2383>
                    var_166 = wp::extract(var_159, var_163);
                    wp::array_store(var_actuator_moment_out, var_0, var_164, var_166);
                    // sparseid = rowadr + i                                                      <L 2381>
                    var_168 = wp::add(var_162, var_167);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2382>
                    var_169 = wp::add(var_30, var_167);
                    wp::array_store(var_moment_colind_out, var_0, var_168, var_169);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2383>
                    var_170 = wp::extract(var_159, var_167);
                    wp::array_store(var_actuator_moment_out, var_0, var_168, var_170);
                    // sparseid = rowadr + i                                                      <L 2381>
                    var_172 = wp::add(var_162, var_171);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2382>
                    var_173 = wp::add(var_30, var_171);
                    wp::array_store(var_moment_colind_out, var_0, var_172, var_173);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2383>
                    var_174 = wp::extract(var_159, var_171);
                    wp::array_store(var_actuator_moment_out, var_0, var_172, var_174);
                }
                if (!var_134) {
                    // elif jnt_typ == JointType.SLIDE or jnt_typ == JointType.HINGE:             <L 2384>
                    var_177 = (var_24 == var_176);
                    var_175 = var_177;
                    if (!var_175) {
                        var_179 = (var_24 == var_178);
                        var_175 = var_175 || var_179;
                    }
                    if (var_175) {
                        // actuator_length_out[worldid, actid] = qpos[qadr] * gear[0]             <L 2385>
                        var_180 = wp::address(var_18, var_27);
                        var_182 = wp::extract(var_11, var_181);
                        var_184 = wp::load(var_180);
                        var_183 = wp::mul(var_184, var_182);
                        wp::array_store(var_actuator_length_out, var_0, var_1, var_183);
                        // nnz = 1                                                                <L 2387>
                        // moment_rownnz_out[worldid, actid] = nnz                                <L 2388>
                        wp::array_store(var_moment_rownnz_out, var_0, var_1, var_185);
                        // rowadr = wp.atomic_add(moment_nnz, worldid, nnz)                       <L 2389>
                        var_186 = wp::atomic_add(var_moment_nnz, var_0, var_185);
                        // moment_rowadr_out[worldid, actid] = rowadr                             <L 2390>
                        wp::array_store(var_moment_rowadr_out, var_0, var_1, var_186);
                        // moment_colind_out[worldid, rowadr] = vadr                              <L 2391>
                        wp::array_store(var_moment_colind_out, var_0, var_186, var_30);
                        // actuator_moment_out[worldid, rowadr] = gear[0]                         <L 2392>
                        var_188 = wp::extract(var_11, var_187);
                        wp::array_store(var_actuator_moment_out, var_0, var_186, var_188);
                    }
                    if (!var_175) {
                        // wp.printf("unrecognized joint type")                                   <L 2394>
                        printf(var_189);
                    }
                }
                var_190 = wp::where(var_134, var_161, var_185);
                var_191 = wp::where(var_134, var_162, var_186);
            }
            var_192 = wp::where(var_33, var_36, var_191);
            var_193 = wp::where(var_33, var_82, var_157);
            var_194 = wp::where(var_33, var_84, var_159);
        }
        if (!var_13) {
            // elif trntype == TrnType.SLIDERCRANK:                                               <L 2395>
            var_196 = (var_3 == var_195);
            if (var_196) {
                // trnid = actuator_trnid[actid]                                                  <L 2397>
                var_197 = wp::address(var_actuator_trnid, var_1);
                var_199 = wp::load(var_197);
                var_198 = wp::copy(var_199);
                // id = trnid[0]                                                                  <L 2398>
                var_201 = wp::extract(var_198, var_200);
                // idslider = trnid[1]                                                            <L 2399>
                var_203 = wp::extract(var_198, var_202);
                // gear0 = gear[0]                                                                <L 2400>
                var_205 = wp::extract(var_11, var_204);
                // rod = actuator_cranklength[worldid % actuator_cranklength.shape[0], actid]       <L 2401>
                var_206 = &(var_actuator_cranklength.shape);
                var_209 = wp::load(var_206);
                var_208 = wp::extract(var_209, var_207);
                var_210 = wp::mod(var_0, var_208);
                var_211 = wp::address(var_actuator_cranklength, var_210, var_1);
                var_213 = wp::load(var_211);
                var_212 = wp::copy(var_213);
                // site_xmat = site_xmat_in[worldid, idslider]                                    <L 2402>
                var_214 = wp::address(var_site_xmat_in, var_0, var_203);
                var_216 = wp::load(var_214);
                var_215 = wp::copy(var_216);
                // axis = wp.vec3(site_xmat[0, 2], site_xmat[1, 2], site_xmat[2, 2])              <L 2403>
                var_219 = wp::extract(var_215, var_217, var_218);
                var_222 = wp::extract(var_215, var_220, var_221);
                var_225 = wp::extract(var_215, var_223, var_224);
                var_226 = wp::vec_t<3, wp::float32>(var_219, var_222, var_225);
                // site_xpos_id = site_xpos_in[worldid, id]                                       <L 2404>
                var_227 = wp::address(var_site_xpos_in, var_0, var_201);
                var_229 = wp::load(var_227);
                var_228 = wp::copy(var_229);
                // site_xpos_idslider = site_xpos_in[worldid, idslider]                           <L 2405>
                var_230 = wp::address(var_site_xpos_in, var_0, var_203);
                var_232 = wp::load(var_230);
                var_231 = wp::copy(var_232);
                // vec = site_xpos_id - site_xpos_idslider                                        <L 2406>
                var_233 = wp::sub(var_228, var_231);
                // av = wp.dot(vec, axis)                                                         <L 2410>
                var_234 = wp::dot(var_233, var_226);
                // det = av * av + rod * rod - wp.dot(vec, vec)                                   <L 2411>
                var_235 = wp::mul(var_234, var_234);
                var_236 = wp::mul(var_212, var_212);
                var_237 = wp::add(var_235, var_236);
                var_238 = wp::dot(var_233, var_233);
                var_239 = wp::sub(var_237, var_238);
                // ok = 1                                                                         <L 2412>
                // if det <= 0.0:                                                                 <L 2413>
                var_242 = (var_239 <= var_241);
                if (var_242) {
                    // ok = 0                                                                     <L 2414>
                    // sdet = 0.0                                                                 <L 2415>
                    // length = av                                                                <L 2416>
                    var_245 = wp::copy(var_234);
                }
                if (!var_242) {
                    // sdet = wp.sqrt(det)                                                        <L 2418>
                    var_246 = wp::sqrt(var_239);
                    // length = av - sdet                                                         <L 2419>
                    var_247 = wp::sub(var_234, var_246);
                }
                var_248 = wp::where(var_242, var_243, var_240);
                var_249 = wp::where(var_242, var_244, var_246);
                var_250 = wp::where(var_242, var_245, var_247);
                // actuator_length_out[worldid, actid] = length * gear0                           <L 2421>
                var_251 = wp::mul(var_250, var_205);
                wp::array_store(var_actuator_length_out, var_0, var_1, var_251);
                // if ok == 1:                                                                    <L 2424>
                var_253 = (var_248 == var_252);
                if (var_253) {
                    // scale = 1.0 - math.safe_div(av, sdet)                                      <L 2425>
                    var_255 = safe_div_0(var_234, var_249);
                    var_256 = wp::sub(var_254, var_255);
                    // dldv = axis * scale + math.safe_div(vec, sdet)                             <L 2426>
                    var_257 = wp::mul(var_226, var_256);
                    var_258 = safe_div_0(var_233, var_249);
                    var_259 = wp::add(var_257, var_258);
                    // dlda = vec * scale                                                         <L 2427>
                    var_260 = wp::mul(var_233, var_256);
                }
                if (!var_253) {
                    // dldv = axis                                                                <L 2429>
                    var_261 = wp::copy(var_226);
                    // dlda = vec                                                                 <L 2430>
                    var_262 = wp::copy(var_233);
                }
                var_263 = wp::where(var_253, var_259, var_261);
                var_264 = wp::where(var_253, var_260, var_262);
                // b1 = body_weldid[site_bodyid[id]]                                              <L 2433>
                var_265 = wp::address(var_site_bodyid, var_201);
                var_267 = wp::load(var_265);
                var_266 = wp::address(var_body_weldid, var_267);
                var_269 = wp::load(var_266);
                var_268 = wp::copy(var_269);
                // b2 = body_weldid[site_bodyid[idslider]]                                        <L 2434>
                var_270 = wp::address(var_site_bodyid, var_203);
                var_272 = wp::load(var_270);
                var_271 = wp::address(var_body_weldid, var_272);
                var_274 = wp::load(var_271);
                var_273 = wp::copy(var_274);
                // da1_init = int(-1)                                                             <L 2435>
                var_276 = wp::int(var_275);
                // da2_init = int(-1)                                                             <L 2436>
                var_278 = wp::int(var_277);
                // if b1 > 0:                                                                     <L 2437>
                var_280 = (var_268 > var_279);
                if (var_280) {
                    // da1_init = body_dofadr[b1] + body_dofnum[b1] - 1                           <L 2438>
                    var_281 = wp::address(var_body_dofadr, var_268);
                    var_282 = wp::address(var_body_dofnum, var_268);
                    var_284 = wp::load(var_281);
                    var_285 = wp::load(var_282);
                    var_283 = wp::add(var_284, var_285);
                    var_287 = wp::sub(var_283, var_286);
                }
                var_288 = wp::where(var_280, var_287, var_276);
                // if b2 > 0:                                                                     <L 2439>
                var_290 = (var_273 > var_289);
                if (var_290) {
                    // da2_init = body_dofadr[b2] + body_dofnum[b2] - 1                           <L 2440>
                    var_291 = wp::address(var_body_dofadr, var_273);
                    var_292 = wp::address(var_body_dofnum, var_273);
                    var_294 = wp::load(var_291);
                    var_295 = wp::load(var_292);
                    var_293 = wp::add(var_294, var_295);
                    var_297 = wp::sub(var_293, var_296);
                }
                var_298 = wp::where(var_290, var_297, var_278);
                // da1 = da1_init                                                                 <L 2442>
                var_299 = wp::copy(var_288);
                // da2 = da2_init                                                                 <L 2443>
                var_300 = wp::copy(var_298);
                // ndof = int(0)                                                                  <L 2444>
                var_302 = wp::int(var_301);
                // while da1 >= 0 or da2 >= 0:                                                    <L 2445>
        start_while_0:;
                var_305 = (var_299 >= var_304);
                var_303 = var_305;
                if (!var_303) {
                    var_307 = (var_300 >= var_306);
                    var_303 = var_303 || var_307;
                }
        if ((var_303) == false) goto end_while_0;
                    // da = wp.max(da1, da2)                                                      <L 2446>
                    var_308 = wp::max(var_299, var_300);
                    // ndof += 1                                                                  <L 2447>
                    var_310 = wp::add(var_302, var_309);
                    // if da1 == da:                                                              <L 2448>
                    var_311 = (var_299 == var_308);
                    if (var_311) {
                        // da1 = dof_parentid[da1]                                                <L 2449>
                        var_312 = wp::address(var_dof_parentid, var_299);
                        var_314 = wp::load(var_312);
                        var_313 = wp::copy(var_314);
                    }
                    var_315 = wp::where(var_311, var_313, var_299);
                    // if da2 == da:                                                              <L 2450>
                    var_316 = (var_300 == var_308);
                    if (var_316) {
                        // da2 = dof_parentid[da2]                                                <L 2451>
                        var_317 = wp::address(var_dof_parentid, var_300);
                        var_319 = wp::load(var_317);
                        var_318 = wp::copy(var_319);
                    }
                    var_320 = wp::where(var_316, var_318, var_300);
                    wp::assign(var_299, var_315);
                    wp::assign(var_300, var_320);
                    wp::assign(var_302, var_310);
        goto start_while_0;
        end_while_0:;
                // moment_rownnz_out[worldid, actid] = ndof                                       <L 2453>
                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_302);
                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)                              <L 2454>
                var_321 = wp::atomic_add(var_moment_nnz, var_0, var_302);
                // moment_rowadr_out[worldid, actid] = rowadr                                     <L 2455>
                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_321);
                // da1 = da1_init                                                                 <L 2458>
                var_322 = wp::copy(var_288);
                // da2 = da2_init                                                                 <L 2459>
                var_323 = wp::copy(var_298);
                // ptr = ndof - 1                                                                 <L 2461>
                var_325 = wp::sub(var_302, var_324);
                // while da1 >= 0 or da2 >= 0:                                                    <L 2462>
        start_while_2:;
                var_328 = (var_322 >= var_327);
                var_326 = var_328;
                if (!var_326) {
                    var_330 = (var_323 >= var_329);
                    var_326 = var_326 || var_330;
                }
        if ((var_326) == false) goto end_while_2;
                    // da = wp.max(da1, da2)                                                      <L 2463>
                    var_331 = wp::max(var_322, var_323);
                    // jacp, jacr = support.jac_dof(                                              <L 2466>
                    // body_parentid,                                                             <L 2467>
                    // body_rootid,                                                               <L 2468>
                    // dof_bodyid,                                                                <L 2469>
                    // body_isdofancestor,                                                        <L 2470>
                    // subtree_com_in,                                                            <L 2471>
                    // cdof_in,                                                                   <L 2472>
                    // site_xpos_idslider,                                                        <L 2473>
                    // site_bodyid[idslider],                                                     <L 2474>
                    var_332 = wp::address(var_site_bodyid, var_203);
                    // da,                                                                        <L 2475>
                    // worldid,                                                                   <L 2476>
                    var_335 = wp::load(var_332);
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_231, var_335, var_331, var_0, var_333, var_334);
                    // jacS = jacp                                                                <L 2478>
                    var_336 = wp::copy(var_333);
                    // jacA = wp.cross(jacr, axis)                                                <L 2479>
                    var_337 = wp::cross(var_334, var_226);
                    // jac, _ = support.jac_dof(                                                  <L 2480>
                    // body_parentid,                                                             <L 2481>
                    // body_rootid,                                                               <L 2482>
                    // dof_bodyid,                                                                <L 2483>
                    // body_isdofancestor,                                                        <L 2484>
                    // subtree_com_in,                                                            <L 2485>
                    // cdof_in,                                                                   <L 2486>
                    // site_xpos_id,                                                              <L 2487>
                    // site_bodyid[id],                                                           <L 2488>
                    var_338 = wp::address(var_site_bodyid, var_201);
                    // da,                                                                        <L 2489>
                    // worldid,                                                                   <L 2490>
                    var_341 = wp::load(var_338);
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_228, var_341, var_331, var_0, var_339, var_340);
                    // jac -= jacS                                                                <L 2492>
                    var_342 = wp::sub(var_339, var_336);
                    // moment = wp.dot(dlda, jacA) + wp.dot(dldv, jac)                            <L 2495>
                    var_343 = wp::dot(var_264, var_337);
                    var_344 = wp::dot(var_263, var_342);
                    var_345 = wp::add(var_343, var_344);
                    // sparseid = rowadr + ptr                                                    <L 2496>
                    var_346 = wp::add(var_321, var_325);
                    // moment_colind_out[worldid, sparseid] = da                                  <L 2497>
                    wp::array_store(var_moment_colind_out, var_0, var_346, var_331);
                    // actuator_moment_out[worldid, sparseid] = moment * gear0                    <L 2498>
                    var_347 = wp::mul(var_345, var_205);
                    wp::array_store(var_actuator_moment_out, var_0, var_346, var_347);
                    // ptr -= 1                                                                   <L 2499>
                    var_349 = wp::sub(var_325, var_348);
                    // if da1 == da:                                                              <L 2501>
                    var_350 = (var_322 == var_331);
                    if (var_350) {
                        // da1 = dof_parentid[da1]                                                <L 2502>
                        var_351 = wp::address(var_dof_parentid, var_322);
                        var_353 = wp::load(var_351);
                        var_352 = wp::copy(var_353);
                    }
                    var_354 = wp::where(var_350, var_352, var_322);
                    // if da2 == da:                                                              <L 2503>
                    var_355 = (var_323 == var_331);
                    if (var_355) {
                        // da2 = dof_parentid[da2]                                                <L 2504>
                        var_356 = wp::address(var_dof_parentid, var_323);
                        var_358 = wp::load(var_356);
                        var_357 = wp::copy(var_358);
                    }
                    var_359 = wp::where(var_355, var_357, var_323);
                    wp::assign(var_322, var_354);
                    wp::assign(var_323, var_359);
                    wp::assign(var_308, var_331);
                    wp::assign(var_325, var_349);
        goto start_while_2;
        end_while_2:;
            }
            if (!var_196) {
                // elif trntype == TrnType.TENDON:                                                <L 2505>
                var_361 = (var_3 == var_360);
                if (var_361) {
                    // tenid = actuator_trnid[actid][0]                                           <L 2506>
                    var_362 = wp::address(var_actuator_trnid, var_1);
                    var_365 = wp::load(var_362);
                    var_364 = wp::extract(var_365, var_363);
                    // gear0 = gear[0]                                                            <L 2508>
                    var_367 = wp::extract(var_11, var_366);
                    // actuator_length_out[worldid, actid] = ten_length_in[worldid, tenid] * gear0       <L 2509>
                    var_368 = wp::address(var_ten_length_in, var_0, var_364);
                    var_370 = wp::load(var_368);
                    var_369 = wp::mul(var_370, var_367);
                    wp::array_store(var_actuator_length_out, var_0, var_1, var_369);
                    // rownnz_ten = ten_J_rownnz[tenid]                                           <L 2511>
                    var_371 = wp::address(var_ten_J_rownnz, var_364);
                    var_373 = wp::load(var_371);
                    var_372 = wp::copy(var_373);
                    // rowadr_ten = ten_J_rowadr[tenid]                                           <L 2512>
                    var_374 = wp::address(var_ten_J_rowadr, var_364);
                    var_376 = wp::load(var_374);
                    var_375 = wp::copy(var_376);
                    // rowadr_mom = wp.atomic_add(moment_nnz, worldid, rownnz_ten)                <L 2514>
                    var_377 = wp::atomic_add(var_moment_nnz, var_0, var_372);
                    // moment_rownnz_out[worldid, actid] = rownnz_ten                             <L 2515>
                    wp::array_store(var_moment_rownnz_out, var_0, var_1, var_372);
                    // moment_rowadr_out[worldid, actid] = rowadr_mom                             <L 2516>
                    wp::array_store(var_moment_rowadr_out, var_0, var_1, var_377);
                    // for k in range(rownnz_ten):                                                <L 2518>
                    var_378 = wp::range(var_372);
                    start_for_4:;
                        if (iter_cmp(var_378) == 0) goto end_for_4;
                        var_379 = wp::iter_next(var_378);
                        // sparseid_ten = rowadr_ten + k                                          <L 2519>
                        var_380 = wp::add(var_375, var_379);
                        // sparseid_mom = rowadr_mom + k                                          <L 2520>
                        var_381 = wp::add(var_377, var_379);
                        // moment_colind_out[worldid, sparseid_mom] = ten_J_colind[sparseid_ten]       <L 2521>
                        var_382 = wp::address(var_ten_J_colind, var_380);
                        var_383 = wp::load(var_382);
                        wp::array_store(var_moment_colind_out, var_0, var_381, var_383);
                        // actuator_moment_out[worldid, sparseid_mom] = ten_J_in[worldid, sparseid_ten] * gear0       <L 2522>
                        var_384 = wp::address(var_ten_J_in, var_0, var_380);
                        var_386 = wp::load(var_384);
                        var_385 = wp::mul(var_386, var_367);
                        wp::array_store(var_actuator_moment_out, var_0, var_381, var_385);
                        goto start_for_4;
                    end_for_4:;
                }
                if (!var_361) {
                    // elif trntype == TrnType.BODY:                                              <L 2523>
                    var_388 = (var_3 == var_387);
                    if (var_388) {
                        // actuator_length_out[worldid, actid] = 0.0                              <L 2525>
                        wp::array_store(var_actuator_length_out, var_0, var_1, var_389);
                        // rowadr = wp.atomic_add(moment_nnz, worldid, nv)                        <L 2528>
                        var_390 = wp::atomic_add(var_moment_nnz, var_0, var_nv);
                        // moment_rownnz_out[worldid, actid] = nv                                 <L 2529>
                        wp::array_store(var_moment_rownnz_out, var_0, var_1, var_nv);
                        // moment_rowadr_out[worldid, actid] = rowadr                             <L 2530>
                        wp::array_store(var_moment_rowadr_out, var_0, var_1, var_390);
                        // for i in range(nv):                                                    <L 2531>
                        var_391 = wp::range(var_nv);
                        start_for_6:;
                            if (iter_cmp(var_391) == 0) goto end_for_6;
                            var_392 = wp::iter_next(var_391);
                            // sparseid = rowadr + i                                              <L 2532>
                            var_393 = wp::add(var_390, var_392);
                            // moment_colind_out[worldid, sparseid] = i                           <L 2533>
                            wp::array_store(var_moment_colind_out, var_0, var_393, var_392);
                            // actuator_moment_out[worldid, sparseid] = 0.0                       <L 2534>
                            wp::array_store(var_actuator_moment_out, var_0, var_393, var_394);
                            goto start_for_6;
                        end_for_6:;
                    }
                    if (!var_388) {
                        // elif trntype == TrnType.SITE:                                          <L 2537>
                        var_396 = (var_3 == var_395);
                        if (var_396) {
                            // trnid = actuator_trnid[actid]                                      <L 2538>
                            var_397 = wp::address(var_actuator_trnid, var_1);
                            var_399 = wp::load(var_397);
                            var_398 = wp::copy(var_399);
                            // siteid = trnid[0]                                                  <L 2539>
                            var_401 = wp::extract(var_398, var_400);
                            // refid = trnid[1]                                                   <L 2540>
                            var_403 = wp::extract(var_398, var_402);
                            // gear = actuator_gear[actuator_gear_id, actid]                      <L 2542>
                            var_404 = wp::address(var_actuator_gear, var_9, var_1);
                            var_406 = wp::load(var_404);
                            var_405 = wp::copy(var_406);
                            // site_quat_id = worldid % site_quat.shape[0]                        <L 2543>
                            var_407 = &(var_site_quat.shape);
                            var_410 = wp::load(var_407);
                            var_409 = wp::extract(var_410, var_408);
                            var_411 = wp::mod(var_0, var_409);
                            // gear_translation = wp.spatial_top(gear)                            <L 2544>
                            var_412 = wp::spatial_top(var_405);
                            // gear_rotational = wp.spatial_bottom(gear)                          <L 2545>
                            var_413 = wp::spatial_bottom(var_405);
                            // if refid == -1:                                                    <L 2548>
                            var_415 = (var_403 == var_414);
                            if (var_415) {
                                // site_xmat = site_xmat_in[worldid, siteid]                      <L 2550>
                                var_416 = wp::address(var_site_xmat_in, var_0, var_401);
                                var_418 = wp::load(var_416);
                                var_417 = wp::copy(var_418);
                                // wrench_translation = site_xmat @ gear_translation              <L 2551>
                                var_419 = wp::mul(var_417, var_412);
                                // wrench_rotation = site_xmat @ gear_rotational                  <L 2552>
                                var_420 = wp::mul(var_417, var_413);
                                // b1 = body_weldid[site_bodyid[siteid]]                          <L 2555>
                                var_421 = wp::address(var_site_bodyid, var_401);
                                var_423 = wp::load(var_421);
                                var_422 = wp::address(var_body_weldid, var_423);
                                var_425 = wp::load(var_422);
                                var_424 = wp::copy(var_425);
                                // da_init = int(-1)                                              <L 2556>
                                var_427 = wp::int(var_426);
                                // if b1 > 0:                                                     <L 2557>
                                var_429 = (var_424 > var_428);
                                if (var_429) {
                                    // da_init = body_dofadr[b1] + body_dofnum[b1] - 1            <L 2558>
                                    var_430 = wp::address(var_body_dofadr, var_424);
                                    var_431 = wp::address(var_body_dofnum, var_424);
                                    var_433 = wp::load(var_430);
                                    var_434 = wp::load(var_431);
                                    var_432 = wp::add(var_433, var_434);
                                    var_436 = wp::sub(var_432, var_435);
                                }
                                var_437 = wp::where(var_429, var_436, var_427);
                                // da = da_init                                                   <L 2560>
                                var_438 = wp::copy(var_437);
                                // ndof = int(0)                                                  <L 2561>
                                var_440 = wp::int(var_439);
                                // while da >= 0:                                                 <L 2562>
        start_while_8:;
                                var_442 = (var_438 >= var_441);
        if ((var_442) == false) goto end_while_8;
                                    // ndof += 1                                                  <L 2563>
                                    var_444 = wp::add(var_440, var_443);
                                    // da = dof_parentid[da]                                      <L 2564>
                                    var_445 = wp::address(var_dof_parentid, var_438);
                                    var_447 = wp::load(var_445);
                                    var_446 = wp::copy(var_447);
                                    wp::assign(var_438, var_446);
                                    wp::assign(var_440, var_444);
        goto start_while_8;
        end_while_8:;
                                // moment_rownnz_out[worldid, actid] = ndof                       <L 2566>
                                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_440);
                                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)              <L 2567>
                                var_448 = wp::atomic_add(var_moment_nnz, var_0, var_440);
                                // moment_rowadr_out[worldid, actid] = rowadr                     <L 2568>
                                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_448);
                                // actuator_length_out[worldid, actid] = 0.0                      <L 2569>
                                wp::array_store(var_actuator_length_out, var_0, var_1, var_449);
                                // da = da_init                                                   <L 2572>
                                var_450 = wp::copy(var_437);
                                // ptr = ndof - 1                                                 <L 2573>
                                var_452 = wp::sub(var_440, var_451);
                                // while da >= 0:                                                 <L 2574>
        start_while_10:;
                                var_454 = (var_450 >= var_453);
        if ((var_454) == false) goto end_while_10;
                                    // jacp, jacr = support.jac_dof(                              <L 2575>
                                    // body_parentid,                                             <L 2576>
                                    // body_rootid,                                               <L 2577>
                                    // dof_bodyid,                                                <L 2578>
                                    // body_isdofancestor,                                        <L 2579>
                                    // subtree_com_in,                                            <L 2580>
                                    // cdof_in,                                                   <L 2581>
                                    // site_xpos_in[worldid, siteid],                             <L 2582>
                                    var_455 = wp::address(var_site_xpos_in, var_0, var_401);
                                    // site_bodyid[siteid],                                       <L 2583>
                                    var_456 = wp::address(var_site_bodyid, var_401);
                                    // da,                                                        <L 2584>
                                    // worldid,                                                   <L 2585>
                                    var_459 = wp::load(var_455);
                                    var_460 = wp::load(var_456);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_459, var_460, var_450, var_0, var_457, var_458);
                                    // moment = wp.dot(jacp, wrench_translation) + wp.dot(jacr, wrench_rotation)       <L 2587>
                                    var_461 = wp::dot(var_457, var_419);
                                    var_462 = wp::dot(var_458, var_420);
                                    var_463 = wp::add(var_461, var_462);
                                    // sparseid = rowadr + ptr                                    <L 2588>
                                    var_464 = wp::add(var_448, var_452);
                                    // moment_colind_out[worldid, sparseid] = da                  <L 2589>
                                    wp::array_store(var_moment_colind_out, var_0, var_464, var_450);
                                    // actuator_moment_out[worldid, sparseid] = moment            <L 2590>
                                    wp::array_store(var_actuator_moment_out, var_0, var_464, var_463);
                                    // ptr -= 1                                                   <L 2591>
                                    var_466 = wp::sub(var_452, var_465);
                                    // da = dof_parentid[da]                                      <L 2592>
                                    var_467 = wp::address(var_dof_parentid, var_450);
                                    var_469 = wp::load(var_467);
                                    var_468 = wp::copy(var_469);
                                    wp::assign(var_450, var_468);
                                    wp::assign(var_452, var_466);
        goto start_while_10;
        end_while_10:;
                            }
                            if (!var_415) {
                                // bodyid = site_bodyid[siteid]                                   <L 2596>
                                var_470 = wp::address(var_site_bodyid, var_401);
                                var_472 = wp::load(var_470);
                                var_471 = wp::copy(var_472);
                                // bodyrefid = site_bodyid[refid]                                 <L 2597>
                                var_473 = wp::address(var_site_bodyid, var_403);
                                var_475 = wp::load(var_473);
                                var_474 = wp::copy(var_475);
                                // b0 = body_weldid[bodyid]                                       <L 2598>
                                var_476 = wp::address(var_body_weldid, var_471);
                                var_478 = wp::load(var_476);
                                var_477 = wp::copy(var_478);
                                // b1 = body_weldid[bodyrefid]                                    <L 2599>
                                var_479 = wp::address(var_body_weldid, var_474);
                                var_481 = wp::load(var_479);
                                var_480 = wp::copy(var_481);
                                // dofadr0 = body_dofadr[b0] + body_dofnum[b0] - 1                <L 2600>
                                var_482 = wp::address(var_body_dofadr, var_477);
                                var_483 = wp::address(var_body_dofnum, var_477);
                                var_485 = wp::load(var_482);
                                var_486 = wp::load(var_483);
                                var_484 = wp::add(var_485, var_486);
                                var_488 = wp::sub(var_484, var_487);
                                // dofadr1 = body_dofadr[b1] + body_dofnum[b1] - 1                <L 2601>
                                var_489 = wp::address(var_body_dofadr, var_480);
                                var_490 = wp::address(var_body_dofnum, var_480);
                                var_492 = wp::load(var_489);
                                var_493 = wp::load(var_490);
                                var_491 = wp::add(var_492, var_493);
                                var_495 = wp::sub(var_491, var_494);
                                // dofadr_common = -1                                             <L 2604>
                                // if dofadr0 >= 0 and dofadr1 >= 0:                              <L 2605>
                                var_499 = (var_488 >= var_498);
                                var_497 = var_499;
                                if (var_497) {
                                    var_501 = (var_495 >= var_500);
                                    var_497 = var_497 && var_501;
                                }
                                if (var_497) {
                                    // while dofadr0 != dofadr1:                                  <L 2607>
        start_while_12:;
                                    var_502 = (var_488 != var_495);
        if ((var_502) == false) goto end_while_12;
                                        // if dofadr0 < dofadr1:                                  <L 2608>
                                        var_503 = (var_488 < var_495);
                                        if (var_503) {
                                            // dofadr1 = dof_parentid[dofadr1]                    <L 2609>
                                            var_504 = wp::address(var_dof_parentid, var_495);
                                            var_506 = wp::load(var_504);
                                            var_505 = wp::copy(var_506);
                                        }
                                        if (!var_503) {
                                            // dofadr0 = dof_parentid[dofadr0]                    <L 2611>
                                            var_507 = wp::address(var_dof_parentid, var_488);
                                            var_509 = wp::load(var_507);
                                            var_508 = wp::copy(var_509);
                                        }
                                        var_510 = wp::where(var_503, var_488, var_508);
                                        var_511 = wp::where(var_503, var_505, var_495);
                                        // if dofadr0 == -1 or dofadr1 == -1:                     <L 2613>
                                        var_514 = (var_510 == var_513);
                                        var_512 = var_514;
                                        if (!var_512) {
                                            var_516 = (var_511 == var_515);
                                            var_512 = var_512 || var_516;
                                        }
                                        if (var_512) {
                                            // break                                              <L 2615>
                                            wp::assign(var_488, var_510);
                                            wp::assign(var_495, var_511);
                                            goto end_while_12;
                                        }
                                        var_517 = wp::where(var_512, var_488, var_510);
                                        var_518 = wp::where(var_512, var_495, var_511);
                                        wp::assign(var_488, var_517);
                                        wp::assign(var_495, var_518);
        goto start_while_12;
        end_while_12:;
                                    // if dofadr0 == dofadr1:                                     <L 2618>
                                    var_519 = (var_488 == var_495);
                                    if (var_519) {
                                        // dofadr_common = dofadr0                                <L 2619>
                                        var_520 = wp::copy(var_488);
                                    }
                                    var_521 = wp::where(var_519, var_520, var_496);
                                }
                                var_522 = wp::where(var_497, var_521, var_496);
                                // translational_transmission = not (gear[0] == 0.0 and gear[1] == 0.0 and gear[2] == 0.0)       <L 2621>
                                var_525 = wp::extract(var_405, var_524);
                                var_527 = (var_525 == var_526);
                                var_523 = var_527;
                                if (var_523) {
                                    var_529 = wp::extract(var_405, var_528);
                                    var_531 = (var_529 == var_530);
                                    var_523 = var_523 && var_531;
                                }
                                if (var_523) {
                                    var_533 = wp::extract(var_405, var_532);
                                    var_535 = (var_533 == var_534);
                                    var_523 = var_523 && var_535;
                                }
                                var_536 = wp::unot(var_523);
                                // rotational_transmission = not (gear[3] == 0.0 and gear[4] == 0.0 and gear[5] == 0.0)       <L 2622>
                                var_539 = wp::extract(var_405, var_538);
                                var_541 = (var_539 == var_540);
                                var_537 = var_541;
                                if (var_537) {
                                    var_543 = wp::extract(var_405, var_542);
                                    var_545 = (var_543 == var_544);
                                    var_537 = var_537 && var_545;
                                }
                                if (var_537) {
                                    var_547 = wp::extract(var_405, var_546);
                                    var_549 = (var_547 == var_548);
                                    var_537 = var_537 && var_549;
                                }
                                var_550 = wp::unot(var_537);
                                // site_xpos = site_xpos_in[worldid, siteid]                      <L 2624>
                                var_551 = wp::address(var_site_xpos_in, var_0, var_401);
                                var_553 = wp::load(var_551);
                                var_552 = wp::copy(var_553);
                                // ref_xpos = site_xpos_in[worldid, refid]                        <L 2625>
                                var_554 = wp::address(var_site_xpos_in, var_0, var_403);
                                var_556 = wp::load(var_554);
                                var_555 = wp::copy(var_556);
                                // ref_xmat = site_xmat_in[worldid, refid]                        <L 2626>
                                var_557 = wp::address(var_site_xmat_in, var_0, var_403);
                                var_559 = wp::load(var_557);
                                var_558 = wp::copy(var_559);
                                // length = float(0.0)                                            <L 2628>
                                var_561 = wp::float(var_560);
                                // if translational_transmission:                                 <L 2630>
                                if (var_536) {
                                    // vec = wp.transpose(ref_xmat) @ (site_xpos - ref_xpos)       <L 2632>
                                    var_562 = wp::transpose(var_558);
                                    var_563 = wp::sub(var_552, var_555);
                                    var_564 = wp::mul(var_562, var_563);
                                    // length += wp.dot(vec, gear_translation)                    <L 2633>
                                    var_565 = wp::dot(var_564, var_412);
                                    var_566 = wp::add(var_561, var_565);
                                    // wrench_translation = ref_xmat @ gear_translation           <L 2635>
                                    var_567 = wp::mul(var_558, var_412);
                                }
                                var_568 = wp::where(var_536, var_566, var_561);
                                // if rotational_transmission:                                    <L 2637>
                                if (var_550) {
                                    // quat = math.mul_quat(site_quat[site_quat_id, siteid], xquat_in[worldid, bodyid])       <L 2639>
                                    var_569 = wp::address(var_site_quat, var_411, var_401);
                                    var_570 = wp::address(var_xquat_in, var_0, var_471);
                                    var_572 = wp::load(var_569);
                                    var_573 = wp::load(var_570);
                                    var_571 = mul_quat_0(var_572, var_573);
                                    // refquat = math.mul_quat(site_quat[site_quat_id, refid], xquat_in[worldid, bodyrefid])       <L 2640>
                                    var_574 = wp::address(var_site_quat, var_411, var_403);
                                    var_575 = wp::address(var_xquat_in, var_0, var_474);
                                    var_577 = wp::load(var_574);
                                    var_578 = wp::load(var_575);
                                    var_576 = mul_quat_0(var_577, var_578);
                                    // vec = math.quat_sub(quat, refquat)                         <L 2643>
                                    var_579 = quat_sub_0(var_571, var_576);
                                    // length += wp.dot(vec, gear_rotational)                     <L 2644>
                                    var_580 = wp::dot(var_579, var_413);
                                    var_581 = wp::add(var_568, var_580);
                                    // wrench_rotation = ref_xmat @ gear_rotational               <L 2646>
                                    var_582 = wp::mul(var_558, var_413);
                                }
                                var_583 = wp::where(var_550, var_581, var_568);
                                var_584 = wp::where(var_550, var_579, var_564);
                                // actuator_length_out[worldid, actid] = length                   <L 2648>
                                wp::array_store(var_actuator_length_out, var_0, var_1, var_583);
                                // da1_init = int(-1)                                             <L 2651>
                                var_586 = wp::int(var_585);
                                // da2_init = int(-1)                                             <L 2652>
                                var_588 = wp::int(var_587);
                                // if b0 > 0:                                                     <L 2653>
                                var_590 = (var_477 > var_589);
                                if (var_590) {
                                    // da1_init = body_dofadr[b0] + body_dofnum[b0] - 1           <L 2654>
                                    var_591 = wp::address(var_body_dofadr, var_477);
                                    var_592 = wp::address(var_body_dofnum, var_477);
                                    var_594 = wp::load(var_591);
                                    var_595 = wp::load(var_592);
                                    var_593 = wp::add(var_594, var_595);
                                    var_597 = wp::sub(var_593, var_596);
                                }
                                var_598 = wp::where(var_590, var_597, var_586);
                                // if b1 > 0:                                                     <L 2655>
                                var_600 = (var_480 > var_599);
                                if (var_600) {
                                    // da2_init = body_dofadr[b1] + body_dofnum[b1] - 1           <L 2656>
                                    var_601 = wp::address(var_body_dofadr, var_480);
                                    var_602 = wp::address(var_body_dofnum, var_480);
                                    var_604 = wp::load(var_601);
                                    var_605 = wp::load(var_602);
                                    var_603 = wp::add(var_604, var_605);
                                    var_607 = wp::sub(var_603, var_606);
                                }
                                var_608 = wp::where(var_600, var_607, var_588);
                                // da1 = da1_init                                                 <L 2658>
                                var_609 = wp::copy(var_598);
                                // da2 = da2_init                                                 <L 2659>
                                var_610 = wp::copy(var_608);
                                // ndof = int(0)                                                  <L 2660>
                                var_612 = wp::int(var_611);
                                // while da1 >= 0 or da2 >= 0:                                    <L 2661>
        start_while_14:;
                                var_615 = (var_609 >= var_614);
                                var_613 = var_615;
                                if (!var_613) {
                                    var_617 = (var_610 >= var_616);
                                    var_613 = var_613 || var_617;
                                }
        if ((var_613) == false) goto end_while_14;
                                    // da = wp.max(da1, da2)                                      <L 2662>
                                    var_618 = wp::max(var_609, var_610);
                                    // if da1 == da and da2 == da:                                <L 2663>
                                    var_620 = (var_609 == var_618);
                                    var_619 = var_620;
                                    if (var_619) {
                                        var_621 = (var_610 == var_618);
                                        var_619 = var_619 && var_621;
                                    }
                                    if (var_619) {
                                        // break                                                  <L 2664>
                                        goto end_while_14;
                                    }
                                    // ndof += 1                                                  <L 2665>
                                    var_623 = wp::add(var_612, var_622);
                                    // if da1 == da:                                              <L 2666>
                                    var_624 = (var_609 == var_618);
                                    if (var_624) {
                                        // da1 = dof_parentid[da1]                                <L 2667>
                                        var_625 = wp::address(var_dof_parentid, var_609);
                                        var_627 = wp::load(var_625);
                                        var_626 = wp::copy(var_627);
                                    }
                                    var_628 = wp::where(var_624, var_626, var_609);
                                    // if da2 == da:                                              <L 2668>
                                    var_629 = (var_610 == var_618);
                                    if (var_629) {
                                        // da2 = dof_parentid[da2]                                <L 2669>
                                        var_630 = wp::address(var_dof_parentid, var_610);
                                        var_632 = wp::load(var_630);
                                        var_631 = wp::copy(var_632);
                                    }
                                    var_633 = wp::where(var_629, var_631, var_610);
                                    wp::assign(var_609, var_628);
                                    wp::assign(var_610, var_633);
                                    wp::assign(var_612, var_623);
        goto start_while_14;
        end_while_14:;
                                // moment_rownnz_out[worldid, actid] = ndof                       <L 2671>
                                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_612);
                                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)              <L 2672>
                                var_634 = wp::atomic_add(var_moment_nnz, var_0, var_612);
                                // moment_rowadr_out[worldid, actid] = rowadr                     <L 2673>
                                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_634);
                                // da1 = da1_init                                                 <L 2676>
                                var_635 = wp::copy(var_598);
                                // da2 = da2_init                                                 <L 2677>
                                var_636 = wp::copy(var_608);
                                // ptr = ndof - 1                                                 <L 2679>
                                var_638 = wp::sub(var_612, var_637);
                                // while da1 >= 0 or da2 >= 0:                                    <L 2680>
        start_while_16:;
                                var_641 = (var_635 >= var_640);
                                var_639 = var_641;
                                if (!var_639) {
                                    var_643 = (var_636 >= var_642);
                                    var_639 = var_639 || var_643;
                                }
        if ((var_639) == false) goto end_while_16;
                                    // da = wp.max(da1, da2)                                      <L 2681>
                                    var_644 = wp::max(var_635, var_636);
                                    // if da1 == da and da2 == da:                                <L 2682>
                                    var_646 = (var_635 == var_644);
                                    var_645 = var_646;
                                    if (var_645) {
                                        var_647 = (var_636 == var_644);
                                        var_645 = var_645 && var_647;
                                    }
                                    if (var_645) {
                                        // break                                                  <L 2683>
                                        wp::assign(var_618, var_644);
                                        goto end_while_16;
                                    }
                                    var_648 = wp::where(var_645, var_618, var_644);
                                    // jacp, jacr = support.jac_dof(                              <L 2685>
                                    // body_parentid,                                             <L 2686>
                                    // body_rootid,                                               <L 2687>
                                    // dof_bodyid,                                                <L 2688>
                                    // body_isdofancestor,                                        <L 2689>
                                    // subtree_com_in,                                            <L 2690>
                                    // cdof_in,                                                   <L 2691>
                                    // site_xpos,                                                 <L 2692>
                                    // site_bodyid[siteid],                                       <L 2693>
                                    var_649 = wp::address(var_site_bodyid, var_401);
                                    // da,                                                        <L 2694>
                                    // worldid,                                                   <L 2695>
                                    var_652 = wp::load(var_649);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_552, var_652, var_648, var_0, var_650, var_651);
                                    // jacpref, jacrref = support.jac_dof(                        <L 2697>
                                    // body_parentid,                                             <L 2698>
                                    // body_rootid,                                               <L 2699>
                                    // dof_bodyid,                                                <L 2700>
                                    // body_isdofancestor,                                        <L 2701>
                                    // subtree_com_in,                                            <L 2702>
                                    // cdof_in,                                                   <L 2703>
                                    // ref_xpos,                                                  <L 2704>
                                    // site_bodyid[refid],                                        <L 2705>
                                    var_653 = wp::address(var_site_bodyid, var_403);
                                    // da,                                                        <L 2706>
                                    // worldid,                                                   <L 2707>
                                    var_656 = wp::load(var_653);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_555, var_656, var_648, var_0, var_654, var_655);
                                    // moment = float(0.0)                                        <L 2710>
                                    var_658 = wp::float(var_657);
                                    // if translational_transmission:                             <L 2711>
                                    if (var_536) {
                                        // moment += wp.dot(jacp - jacpref, wrench_translation)       <L 2712>
                                        var_659 = wp::sub(var_650, var_654);
                                        var_660 = wp::dot(var_659, var_567);
                                        var_661 = wp::add(var_658, var_660);
                                    }
                                    var_662 = wp::where(var_536, var_661, var_658);
                                    // if rotational_transmission:                                <L 2713>
                                    if (var_550) {
                                        // moment += wp.dot(jacr - jacrref, wrench_rotation)       <L 2714>
                                        var_663 = wp::sub(var_651, var_655);
                                        var_664 = wp::dot(var_663, var_582);
                                        var_665 = wp::add(var_662, var_664);
                                    }
                                    var_666 = wp::where(var_550, var_665, var_662);
                                    // sparseid = rowadr + ptr                                    <L 2716>
                                    var_667 = wp::add(var_634, var_638);
                                    // moment_colind_out[worldid, sparseid] = da                  <L 2717>
                                    wp::array_store(var_moment_colind_out, var_0, var_667, var_648);
                                    // actuator_moment_out[worldid, sparseid] = moment            <L 2718>
                                    wp::array_store(var_actuator_moment_out, var_0, var_667, var_666);
                                    // ptr -= 1                                                   <L 2719>
                                    var_669 = wp::sub(var_638, var_668);
                                    // if da1 == da:                                              <L 2721>
                                    var_670 = (var_635 == var_648);
                                    if (var_670) {
                                        // da1 = dof_parentid[da1]                                <L 2722>
                                        var_671 = wp::address(var_dof_parentid, var_635);
                                        var_673 = wp::load(var_671);
                                        var_672 = wp::copy(var_673);
                                    }
                                    var_674 = wp::where(var_670, var_672, var_635);
                                    // if da2 == da:                                              <L 2723>
                                    var_675 = (var_636 == var_648);
                                    if (var_675) {
                                        // da2 = dof_parentid[da2]                                <L 2724>
                                        var_676 = wp::address(var_dof_parentid, var_636);
                                        var_678 = wp::load(var_676);
                                        var_677 = wp::copy(var_678);
                                    }
                                    var_679 = wp::where(var_675, var_677, var_636);
                                    wp::assign(var_635, var_674);
                                    wp::assign(var_636, var_679);
                                    wp::assign(var_618, var_648);
                                    wp::assign(var_638, var_669);
        goto start_while_16;
        end_while_16:;
                            }
                            var_680 = wp::where(var_415, var_419, var_567);
                            var_681 = wp::where(var_415, var_420, var_582);
                            var_682 = wp::where(var_415, var_424, var_480);
                            var_683 = wp::where(var_415, var_450, var_618);
                            var_684 = wp::where(var_415, var_440, var_612);
                            var_685 = wp::where(var_415, var_448, var_634);
                            var_686 = wp::where(var_415, var_452, var_638);
                            var_687 = wp::where(var_415, var_457, var_650);
                            var_688 = wp::where(var_415, var_458, var_651);
                            var_689 = wp::where(var_415, var_463, var_666);
                            var_690 = wp::where(var_415, var_464, var_667);
                        }
                        if (!var_396) {
                            // wp.printf("unhandled transmission type %d\n", trntype)             <L 2726>
                            printf(var_691, var_3);
                        }
                        var_692 = wp::where(var_396, var_405, var_11);
                    }
                    var_693 = wp::where(var_388, var_11, var_692);
                    var_694 = wp::where(var_388, var_390, var_685);
                    var_695 = wp::where(var_388, var_393, var_690);
                }
                var_696 = wp::where(var_361, var_11, var_693);
            }
            var_697 = wp::where(var_196, var_11, var_696);
            var_698 = wp::where(var_196, var_198, var_398);
            var_699 = wp::where(var_196, var_205, var_367);
            var_700 = wp::where(var_196, var_215, var_417);
            var_701 = wp::where(var_196, var_233, var_584);
            var_702 = wp::where(var_196, var_250, var_583);
            var_703 = wp::where(var_196, var_268, var_682);
            var_704 = wp::where(var_196, var_288, var_598);
            var_705 = wp::where(var_196, var_298, var_608);
            var_706 = wp::where(var_196, var_322, var_635);
            var_707 = wp::where(var_196, var_323, var_636);
            var_708 = wp::where(var_196, var_302, var_684);
            var_709 = wp::where(var_196, var_308, var_683);
            var_710 = wp::where(var_196, var_321, var_694);
            var_711 = wp::where(var_196, var_325, var_686);
            var_712 = wp::where(var_196, var_333, var_687);
            var_713 = wp::where(var_196, var_334, var_688);
            var_714 = wp::where(var_196, var_345, var_689);
            var_715 = wp::where(var_196, var_346, var_695);
        }
        var_716 = wp::where(var_13, var_11, var_697);
        var_717 = wp::where(var_13, var_192, var_710);
        var_718 = wp::where(var_13, var_81, var_571);
        var_719 = wp::where(var_13, var_171, var_392);
        var_720 = wp::where(var_13, var_172, var_715);
    }
}



extern "C" __global__ void _count_equality_constraints_54fc2a85_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_ne_connect_out,
    wp::array_t<wp::int32> var_ne_weld_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        bool var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        //---------
        // forward
        // def _count_equality_constraints(                                                       <L 1540>
        // worldid, efcid = wp.tid()                                                              <L 1552>
        builtin_tid2d(var_0, var_1);
        // if efcid >= ne_in[worldid]:                                                            <L 1555>
        var_2 = wp::address(var_ne_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1556>
            continue;
        }
        // eq_id = efc_id_in[worldid, efcid]                                                      <L 1559>
        var_5 = wp::address(var_efc_id_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // eq_constraint_type = eq_type[eq_id]                                                    <L 1560>
        var_8 = wp::address(var_eq_type, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if eq_constraint_type == EqType.CONNECT:                                               <L 1563>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // wp.atomic_add(ne_connect_out, worldid, 1)                                          <L 1564>
            var_14 = wp::atomic_add(var_ne_connect_out, var_0, var_13);
        }
        if (!var_12) {
            // elif eq_constraint_type == EqType.WELD:                                            <L 1565>
            var_16 = (var_9 == var_15);
            if (var_16) {
                // wp.atomic_add(ne_weld_out, worldid, 1)                                         <L 1566>
                var_18 = wp::atomic_add(var_ne_weld_out, var_0, var_17);
            }
        }
    }
}



extern "C" __global__ void _tendon_dot_62db264d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_tendon_adr,
    wp::array_t<wp::int32> var_tendon_num,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::array_t<wp::float32> var_ten_Jdot_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 1.0;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        bool var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 2;
        bool var_54;
        bool var_55;
        bool var_56;
        bool var_57;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::vec_t<6, wp::float32>* var_71;
        wp::vec_t<6, wp::float32> var_72;
        wp::vec_t<6, wp::float32> var_73;
        wp::int32* var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::int32 var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        bool var_84;
        const wp::int32 var_85 = 4;
        bool var_86;
        const wp::int32 var_87 = 5;
        bool var_88;
        wp::int32* var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        wp::vec_t<3, wp::float32>* var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<6, wp::float32>* var_95;
        wp::vec_t<6, wp::float32> var_96;
        wp::vec_t<6, wp::float32> var_97;
        wp::int32* var_98;
        wp::vec_t<3, wp::float32>* var_99;
        wp::int32 var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        bool var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::float32 var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        const wp::float32 var_127 = 1e-15;
        bool var_128;
        wp::vec_t<3, wp::float32> var_129;
        const wp::float32 var_130 = 0.0;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::int32* var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32* var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        const wp::float32 var_139 = 1.0;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 1;
        wp::int32 var_144;
        //---------
        // forward
        // def _tendon_dot(                                                                       <L 1897>
        // worldid, tenid = wp.tid()                                                              <L 1925>
        builtin_tid2d(var_0, var_1);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 1927>
        var_2 = &(var_tendon_armature.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_tendon_armature, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if armature == 0.0:                                                                    <L 1928>
        var_11 = (var_8 == var_10);
        if (var_11) {
            // return                                                                             <L 1929>
            continue;
        }
        // adr = tendon_adr[tenid]                                                                <L 1932>
        var_12 = wp::address(var_tendon_adr, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if wrap_type[adr] == WrapType.JOINT:                                                   <L 1933>
        var_15 = wp::address(var_wrap_type, var_13);
        var_18 = wp::load(var_15);
        var_17 = (var_18 == var_16);
        if (var_17) {
            // return                                                                             <L 1934>
            continue;
        }
        // divisor = float(1.0)                                                                   <L 1937>
        var_20 = wp::float(var_19);
        // num = tendon_num[tenid]                                                                <L 1938>
        var_21 = wp::address(var_tendon_num, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // j = int(0)                                                                             <L 1939>
        var_25 = wp::int(var_24);
        // while j < num - 1:                                                                     <L 1940>
        start_while_2:;
        var_27 = wp::sub(var_22, var_26);
        var_28 = (var_25 < var_27);
        if ((var_28) == false) goto end_while_2;
            // type0 = wrap_type[adr + j + 0]                                                     <L 1942>
            var_29 = wp::add(var_13, var_25);
            var_31 = wp::add(var_29, var_30);
            var_32 = wp::address(var_wrap_type, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // type1 = wrap_type[adr + j + 1]                                                     <L 1943>
            var_35 = wp::add(var_13, var_25);
            var_37 = wp::add(var_35, var_36);
            var_38 = wp::address(var_wrap_type, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // id0 = wrap_objid[adr + j + 0]                                                      <L 1944>
            var_41 = wp::add(var_13, var_25);
            var_43 = wp::add(var_41, var_42);
            var_44 = wp::address(var_wrap_objid, var_43);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // id1 = wrap_objid[adr + j + 1]                                                      <L 1945>
            var_47 = wp::add(var_13, var_25);
            var_49 = wp::add(var_47, var_48);
            var_50 = wp::address(var_wrap_objid, var_49);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // pulley = WrapType.PULLEY                                                           <L 1948>
            // if (type0 == pulley) or (type1 == pulley):                                         <L 1949>
            var_55 = (var_33 == var_53);
            var_54 = var_55;
            if (!var_54) {
                var_56 = (var_39 == var_53);
                var_54 = var_54 || var_56;
            }
            if (var_54) {
                // if type0 == pulley:                                                            <L 1951>
                var_57 = (var_33 == var_53);
                if (var_57) {
                    // divisor = wrap_prm[adr + j]                                                <L 1952>
                    var_58 = wp::add(var_13, var_25);
                    var_59 = wp::address(var_wrap_prm, var_58);
                    var_61 = wp::load(var_59);
                    var_60 = wp::copy(var_61);
                }
                var_62 = wp::where(var_57, var_60, var_20);
                // j += 1                                                                         <L 1954>
                var_64 = wp::add(var_25, var_63);
                // continue                                                                       <L 1955>
                wp::assign(var_20, var_62);
                wp::assign(var_25, var_64);
                goto start_while_2;
            }
            // wpnt0 = site_xpos_in[worldid, id0]                                                 <L 1958>
            var_65 = wp::address(var_site_xpos_in, var_0, var_45);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // wbody0 = site_bodyid[id0]                                                          <L 1960>
            var_68 = wp::address(var_site_bodyid, var_45);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // cvel0 = cvel_in[worldid, wbody0]                                                   <L 1961>
            var_71 = wp::address(var_cvel_in, var_0, var_69);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            // subtree_com0 = subtree_com_in[worldid, body_rootid[wbody0]]                        <L 1962>
            var_74 = wp::address(var_body_rootid, var_69);
            var_76 = wp::load(var_74);
            var_75 = wp::address(var_subtree_com_in, var_0, var_76);
            var_78 = wp::load(var_75);
            var_77 = wp::copy(var_78);
            // offset0 = wpnt0 - subtree_com0                                                     <L 1963>
            var_79 = wp::sub(var_66, var_77);
            // pvel_lin0 = wp.spatial_bottom(cvel0) - wp.cross(offset0, wp.spatial_top(cvel0))       <L 1964>
            var_80 = wp::spatial_bottom(var_72);
            var_81 = wp::spatial_top(var_72);
            var_82 = wp::cross(var_79, var_81);
            var_83 = wp::sub(var_80, var_82);
            // if (type1 == WrapType.SPHERE) or (type1 == WrapType.CYLINDER):                     <L 1967>
            var_86 = (var_39 == var_85);
            var_84 = var_86;
            if (!var_84) {
                var_88 = (var_39 == var_87);
                var_84 = var_84 || var_88;
            }
            if (var_84) {
                // return                                                                         <L 1969>
                continue;
            }
            // wbody1 = site_bodyid[id1]                                                          <L 1972>
            var_89 = wp::address(var_site_bodyid, var_51);
            var_91 = wp::load(var_89);
            var_90 = wp::copy(var_91);
            // wpnt1 = site_xpos_in[worldid, id1]                                                 <L 1973>
            var_92 = wp::address(var_site_xpos_in, var_0, var_51);
            var_94 = wp::load(var_92);
            var_93 = wp::copy(var_94);
            // cvel1 = cvel_in[worldid, wbody1]                                                   <L 1975>
            var_95 = wp::address(var_cvel_in, var_0, var_90);
            var_97 = wp::load(var_95);
            var_96 = wp::copy(var_97);
            // subtree_com1 = subtree_com_in[worldid, body_rootid[wbody1]]                        <L 1976>
            var_98 = wp::address(var_body_rootid, var_90);
            var_100 = wp::load(var_98);
            var_99 = wp::address(var_subtree_com_in, var_0, var_100);
            var_102 = wp::load(var_99);
            var_101 = wp::copy(var_102);
            // offset1 = wpnt1 - subtree_com1                                                     <L 1977>
            var_103 = wp::sub(var_93, var_101);
            // pvel_lin1 = wp.spatial_bottom(cvel1) - wp.cross(offset1, wp.spatial_top(cvel1))       <L 1978>
            var_104 = wp::spatial_bottom(var_96);
            var_105 = wp::spatial_top(var_96);
            var_106 = wp::cross(var_103, var_105);
            var_107 = wp::sub(var_104, var_106);
            // if wbody0 != wbody1:                                                               <L 1981>
            var_108 = (var_69 != var_90);
            if (var_108) {
                // dpnt, norm = math.normalize_with_norm(wpnt1 - wpnt0)                           <L 1983>
                var_109 = wp::sub(var_93, var_66);
                normalize_with_norm_0(var_109, var_110, var_111);
                // wvel0 = wp.spatial_bottom(cvel0) - wp.cross(wpnt0 - subtree_com0, wp.spatial_top(cvel0))       <L 1986>
                var_112 = wp::spatial_bottom(var_72);
                var_113 = wp::sub(var_66, var_77);
                var_114 = wp::spatial_top(var_72);
                var_115 = wp::cross(var_113, var_114);
                var_116 = wp::sub(var_112, var_115);
                // wvel1 = wp.spatial_bottom(cvel1) - wp.cross(wpnt1 - subtree_com1, wp.spatial_top(cvel1))       <L 1987>
                var_117 = wp::spatial_bottom(var_96);
                var_118 = wp::sub(var_93, var_101);
                var_119 = wp::spatial_top(var_96);
                var_120 = wp::cross(var_118, var_119);
                var_121 = wp::sub(var_117, var_120);
                // dvel = wvel1 - wvel0                                                           <L 1988>
                var_122 = wp::sub(var_121, var_116);
                // dot = wp.dot(dpnt, dvel)                                                       <L 1989>
                var_123 = wp::dot(var_110, var_122);
                // dvel += dpnt * (-dot)                                                          <L 1990>
                var_124 = wp::neg(var_123);
                var_125 = wp::mul(var_110, var_124);
                var_126 = wp::add(var_122, var_125);
                // if norm > MJ_MINVAL:                                                           <L 1991>
                var_128 = (var_111 > var_127);
                if (var_128) {
                    // dvel /= norm                                                               <L 1992>
                    var_129 = wp::div(var_126, var_111);
                }
                if (!var_128) {
                    // dvel = wp.vec3(0.0)                                                        <L 1994>
                    var_131 = wp::vec_t<3, wp::float32>(var_130);
                }
                var_132 = wp::where(var_128, var_129, var_131);
                // rownnz = ten_J_rownnz[tenid]                                                   <L 1996>
                var_133 = wp::address(var_ten_J_rownnz, var_1);
                var_135 = wp::load(var_133);
                var_134 = wp::copy(var_135);
                // rowadr = ten_J_rowadr[tenid]                                                   <L 1997>
                var_136 = wp::address(var_ten_J_rowadr, var_1);
                var_138 = wp::load(var_136);
                var_137 = wp::copy(var_138);
                // inv_divisor = math.safe_div(float(1.0), divisor)                               <L 1998>
                var_140 = wp::float(var_139);
                var_141 = safe_div_0(var_140, var_20);
                // _accumulate_jac_dot_chain(                                                     <L 2001>
                // body_parentid,                                                                 <L 2002>
                // body_dofnum,                                                                   <L 2003>
                // body_dofadr,                                                                   <L 2004>
                // jnt_type,                                                                      <L 2005>
                // jnt_dofadr,                                                                    <L 2006>
                // dof_jntid,                                                                     <L 2007>
                // ten_J_colind,                                                                  <L 2008>
                // cdof_in,                                                                       <L 2009>
                // cvel_in,                                                                       <L 2010>
                // cdof_dot_in,                                                                   <L 2011>
                // offset0,                                                                       <L 2012>
                // pvel_lin0,                                                                     <L 2013>
                // dpnt,                                                                          <L 2014>
                // dvel,                                                                          <L 2015>
                // wbody0,                                                                        <L 2016>
                // rowadr,                                                                        <L 2017>
                // rownnz,                                                                        <L 2018>
                // -inv_divisor,                                                                  <L 2019>
                var_142 = wp::neg(var_141);
                // worldid,                                                                       <L 2020>
                // ten_Jdot_out,                                                                  <L 2021>
                _accumulate_jac_dot_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_jnt_type, var_jnt_dofadr, var_dof_jntid, var_ten_J_colind, var_cdof_in, var_cvel_in, var_cdof_dot_in, var_79, var_83, var_110, var_132, var_69, var_137, var_134, var_142, var_0, var_ten_Jdot_out);
                // _accumulate_jac_dot_chain(                                                     <L 2023>
                // body_parentid,                                                                 <L 2024>
                // body_dofnum,                                                                   <L 2025>
                // body_dofadr,                                                                   <L 2026>
                // jnt_type,                                                                      <L 2027>
                // jnt_dofadr,                                                                    <L 2028>
                // dof_jntid,                                                                     <L 2029>
                // ten_J_colind,                                                                  <L 2030>
                // cdof_in,                                                                       <L 2031>
                // cvel_in,                                                                       <L 2032>
                // cdof_dot_in,                                                                   <L 2033>
                // offset1,                                                                       <L 2034>
                // pvel_lin1,                                                                     <L 2035>
                // dpnt,                                                                          <L 2036>
                // dvel,                                                                          <L 2037>
                // wbody1,                                                                        <L 2038>
                // rowadr,                                                                        <L 2039>
                // rownnz,                                                                        <L 2040>
                // inv_divisor,                                                                   <L 2041>
                // worldid,                                                                       <L 2042>
                // ten_Jdot_out,                                                                  <L 2043>
                _accumulate_jac_dot_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_jnt_type, var_jnt_dofadr, var_dof_jntid, var_ten_J_colind, var_cdof_in, var_cvel_in, var_cdof_dot_in, var_103, var_107, var_110, var_132, var_90, var_137, var_134, var_141, var_0, var_ten_Jdot_out);
            }
            // j += 1                                                                             <L 2047>
            var_144 = wp::add(var_25, var_143);
            wp::assign(var_25, var_144);
        goto start_while_2;
        end_while_2:;
    }
}



extern "C" __global__ void _flex_vertices_68b379d4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::vec_t<3, wp::int32>> var_flex_cellnum,
    wp::array_t<wp::int32> var_flex_nodeadr,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_vertnum,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_vert,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_vert0,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::int32>* var_20;
        wp::vec_t<3, wp::int32> var_21;
        wp::vec_t<3, wp::int32> var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        const wp::int32 var_27 = 2;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::int32 var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        const wp::int32 var_39 = 1;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        const wp::int32 var_49 = 2;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::int32 var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::float32 var_65 = 0.0;
        const wp::float32 var_66 = 1.0;
        wp::float32 var_67;
        const wp::int32 var_68 = 1;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::float32 var_74 = 0.0;
        const wp::float32 var_75 = 1.0;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        const wp::float32 var_84 = 1.0;
        wp::float32 var_85;
        wp::vec_t<3, wp::float32> var_86;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        const wp::int32 var_89 = 1;
        wp::int32 var_90;
        wp::int32* var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        const wp::float32 var_94 = 0.0;
        const wp::float32 var_95 = 0.0;
        const wp::float32 var_96 = 0.0;
        wp::vec_t<3, wp::float32> var_97;
        const wp::int32 var_98 = 0;
        const wp::int32 var_99 = 0;
        const wp::int32 var_100 = 0;
        const wp::int32 var_101 = 4;
        wp::int32 var_102;
        const wp::int32 var_103 = 2;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::vec_t<3, wp::float32>* var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        const wp::int32 var_121 = 1;
        const wp::int32 var_122 = 4;
        wp::int32 var_123;
        const wp::int32 var_124 = 2;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::float32 var_128;
        wp::int32 var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::vec_t<3, wp::float32>* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        const wp::int32 var_142 = 1;
        const wp::int32 var_143 = 0;
        const wp::int32 var_144 = 4;
        wp::int32 var_145;
        const wp::int32 var_146 = 2;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::int32 var_149;
        wp::float32 var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::vec_t<3, wp::float32>* var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::vec_t<3, wp::float32> var_163;
        const wp::int32 var_164 = 1;
        const wp::int32 var_165 = 4;
        wp::int32 var_166;
        const wp::int32 var_167 = 2;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::float32 var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        wp::vec_t<3, wp::float32>* var_181;
        wp::vec_t<3, wp::float32> var_182;
        wp::vec_t<3, wp::float32> var_183;
        wp::vec_t<3, wp::float32> var_184;
        const wp::int32 var_185 = 1;
        const wp::int32 var_186 = 0;
        const wp::int32 var_187 = 0;
        const wp::int32 var_188 = 4;
        wp::int32 var_189;
        const wp::int32 var_190 = 2;
        wp::int32 var_191;
        wp::int32 var_192;
        wp::int32 var_193;
        wp::float32 var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        wp::int32 var_197;
        wp::int32 var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        wp::int32 var_201;
        wp::int32 var_202;
        wp::int32 var_203;
        wp::vec_t<3, wp::float32>* var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::vec_t<3, wp::float32> var_206;
        wp::vec_t<3, wp::float32> var_207;
        const wp::int32 var_208 = 1;
        const wp::int32 var_209 = 4;
        wp::int32 var_210;
        const wp::int32 var_211 = 2;
        wp::int32 var_212;
        wp::int32 var_213;
        wp::int32 var_214;
        wp::float32 var_215;
        wp::int32 var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        wp::int32 var_219;
        wp::int32 var_220;
        wp::int32 var_221;
        wp::int32 var_222;
        wp::int32 var_223;
        wp::int32 var_224;
        wp::vec_t<3, wp::float32>* var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        const wp::int32 var_229 = 1;
        const wp::int32 var_230 = 0;
        const wp::int32 var_231 = 4;
        wp::int32 var_232;
        const wp::int32 var_233 = 2;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::float32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        wp::int32 var_241;
        wp::int32 var_242;
        wp::int32 var_243;
        wp::int32 var_244;
        wp::int32 var_245;
        wp::int32 var_246;
        wp::vec_t<3, wp::float32>* var_247;
        wp::vec_t<3, wp::float32> var_248;
        wp::vec_t<3, wp::float32> var_249;
        wp::vec_t<3, wp::float32> var_250;
        const wp::int32 var_251 = 1;
        const wp::int32 var_252 = 4;
        wp::int32 var_253;
        const wp::int32 var_254 = 2;
        wp::int32 var_255;
        wp::int32 var_256;
        wp::int32 var_257;
        wp::float32 var_258;
        wp::int32 var_259;
        wp::int32 var_260;
        wp::int32 var_261;
        wp::int32 var_262;
        wp::int32 var_263;
        wp::int32 var_264;
        wp::int32 var_265;
        wp::int32 var_266;
        wp::int32 var_267;
        wp::vec_t<3, wp::float32>* var_268;
        wp::vec_t<3, wp::float32> var_269;
        wp::vec_t<3, wp::float32> var_270;
        wp::vec_t<3, wp::float32> var_271;
        wp::int32* var_272;
        wp::int32 var_273;
        wp::int32 var_274;
        wp::vec_t<3, wp::float32>* var_275;
        wp::vec_t<3, wp::float32> var_276;
        wp::vec_t<3, wp::float32> var_277;
        bool* var_278;
        bool var_279;
        bool var_280;
        bool var_281;
        wp::mat_t<3, 3, wp::float32>* var_282;
        wp::mat_t<3, 3, wp::float32> var_283;
        wp::mat_t<3, 3, wp::float32> var_284;
        wp::vec_t<3, wp::float32>* var_285;
        wp::vec_t<3, wp::float32> var_286;
        wp::vec_t<3, wp::float32> var_287;
        wp::vec_t<3, wp::float32> var_288;
        wp::vec_t<3, wp::float32> var_289;
        bool var_290;
        //---------
        // forward
        // def _flex_vertices(                                                                    <L 230>
        // worldid, vertid = wp.tid()                                                             <L 249>
        builtin_tid2d(var_0, var_1);
        // for f in range(nflex):                                                                 <L 251>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = vertid - flex_vertadr[f]                                                   <L 252>
            var_4 = wp::address(var_flex_vertadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_vertnum[f]:                                         <L 253>
            var_9 = (var_5 >= var_8);
            var_7 = var_9;
            if (var_7) {
                var_10 = wp::address(var_flex_vertnum, var_3);
                var_12 = wp::load(var_10);
                var_11 = (var_5 < var_12);
                var_7 = var_7 && var_11;
            }
            if (var_7) {
                // break                                                                          <L 254>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // if flex_interp[f] != 0:                                                                <L 256>
        var_13 = wp::address(var_flex_interp, var_3);
        var_16 = wp::load(var_13);
        var_15 = (var_16 != var_14);
        if (var_15) {
            // coord = flex_vert0[vertid]                                                         <L 258>
            var_17 = wp::address(var_flex_vert0, var_1);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // cn = flex_cellnum[f]                                                               <L 259>
            var_20 = wp::address(var_flex_cellnum, var_3);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // cx = cn[0]                                                                         <L 260>
            var_24 = wp::extract(var_21, var_23);
            // cy = cn[1]                                                                         <L 261>
            var_26 = wp::extract(var_21, var_25);
            // cz = cn[2]                                                                         <L 262>
            var_28 = wp::extract(var_21, var_27);
            // ci = wp.min(int(coord[0] * float(cx)), cx - 1)                                     <L 265>
            var_30 = wp::extract(var_18, var_29);
            var_31 = wp::float(var_24);
            var_32 = wp::mul(var_30, var_31);
            var_33 = wp::int(var_32);
            var_35 = wp::sub(var_24, var_34);
            var_36 = wp::min(var_33, var_35);
            // ci = wp.max(ci, 0)                                                                 <L 266>
            var_38 = wp::max(var_36, var_37);
            // cj = wp.min(int(coord[1] * float(cy)), cy - 1)                                     <L 267>
            var_40 = wp::extract(var_18, var_39);
            var_41 = wp::float(var_26);
            var_42 = wp::mul(var_40, var_41);
            var_43 = wp::int(var_42);
            var_45 = wp::sub(var_26, var_44);
            var_46 = wp::min(var_43, var_45);
            // cj = wp.max(cj, 0)                                                                 <L 268>
            var_48 = wp::max(var_46, var_47);
            // ck = wp.min(int(coord[2] * float(cz)), cz - 1)                                     <L 269>
            var_50 = wp::extract(var_18, var_49);
            var_51 = wp::float(var_28);
            var_52 = wp::mul(var_50, var_51);
            var_53 = wp::int(var_52);
            var_55 = wp::sub(var_28, var_54);
            var_56 = wp::min(var_53, var_55);
            // ck = wp.max(ck, 0)                                                                 <L 270>
            var_58 = wp::max(var_56, var_57);
            // local_x = wp.clamp(coord[0] * float(cx) - float(ci), 0.0, 1.0)                     <L 273>
            var_60 = wp::extract(var_18, var_59);
            var_61 = wp::float(var_24);
            var_62 = wp::mul(var_60, var_61);
            var_63 = wp::float(var_38);
            var_64 = wp::sub(var_62, var_63);
            var_67 = wp::clamp(var_64, var_65, var_66);
            // local_y = wp.clamp(coord[1] * float(cy) - float(cj), 0.0, 1.0)                     <L 274>
            var_69 = wp::extract(var_18, var_68);
            var_70 = wp::float(var_26);
            var_71 = wp::mul(var_69, var_70);
            var_72 = wp::float(var_48);
            var_73 = wp::sub(var_71, var_72);
            var_76 = wp::clamp(var_73, var_74, var_75);
            // local_z = wp.clamp(coord[2] * float(cz) - float(ck), 0.0, 1.0)                     <L 275>
            var_78 = wp::extract(var_18, var_77);
            var_79 = wp::float(var_28);
            var_80 = wp::mul(var_78, var_79);
            var_81 = wp::float(var_58);
            var_82 = wp::sub(var_80, var_81);
            var_85 = wp::clamp(var_82, var_83, var_84);
            // local = wp.vec3(local_x, local_y, local_z)                                         <L 276>
            var_86 = wp::vec_t<3, wp::float32>(var_67, var_76, var_85);
            // ny_g = cy + 1                                                                      <L 279>
            var_88 = wp::add(var_26, var_87);
            // nz_g = cz + 1                                                                      <L 280>
            var_90 = wp::add(var_28, var_89);
            // nstart = flex_nodeadr[f]                                                           <L 281>
            var_91 = wp::address(var_flex_nodeadr, var_3);
            var_93 = wp::load(var_91);
            var_92 = wp::copy(var_93);
            // result = wp.vec3(0.0, 0.0, 0.0)                                                    <L 284>
            var_97 = wp::vec_t<3, wp::float32>(var_94, var_95, var_96);
            // for li in range(2):                                                                <L 285>
            // for lj in range(2):                                                                <L 286>
            // for lk in range(2):                                                                <L 287>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_102 = wp::mul(var_98, var_101);
            var_104 = wp::mul(var_99, var_103);
            var_105 = wp::add(var_102, var_104);
            var_106 = wp::add(var_105, var_100);
            var_107 = eval_basis_trilinear_0(var_86, var_106);
            // gi = ci + li                                                                       <L 289>
            var_108 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 290>
            var_109 = wp::add(var_48, var_99);
            // gk = ck + lk                                                                       <L 291>
            var_110 = wp::add(var_58, var_100);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_111 = wp::mul(var_108, var_88);
            var_112 = wp::mul(var_111, var_90);
            var_113 = wp::mul(var_109, var_90);
            var_114 = wp::add(var_112, var_113);
            var_115 = wp::add(var_114, var_110);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_116 = wp::add(var_92, var_115);
            var_117 = wp::address(var_flexnode_xpos_in, var_0, var_116);
            var_119 = wp::load(var_117);
            var_118 = wp::mul(var_107, var_119);
            var_120 = wp::add(var_97, var_118);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_123 = wp::mul(var_98, var_122);
            var_125 = wp::mul(var_99, var_124);
            var_126 = wp::add(var_123, var_125);
            var_127 = wp::add(var_126, var_121);
            var_128 = eval_basis_trilinear_0(var_86, var_127);
            // gi = ci + li                                                                       <L 289>
            var_129 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 290>
            var_130 = wp::add(var_48, var_99);
            // gk = ck + lk                                                                       <L 291>
            var_131 = wp::add(var_58, var_121);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_132 = wp::mul(var_129, var_88);
            var_133 = wp::mul(var_132, var_90);
            var_134 = wp::mul(var_130, var_90);
            var_135 = wp::add(var_133, var_134);
            var_136 = wp::add(var_135, var_131);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_137 = wp::add(var_92, var_136);
            var_138 = wp::address(var_flexnode_xpos_in, var_0, var_137);
            var_140 = wp::load(var_138);
            var_139 = wp::mul(var_128, var_140);
            var_141 = wp::add(var_120, var_139);
            // for lk in range(2):                                                                <L 287>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_145 = wp::mul(var_98, var_144);
            var_147 = wp::mul(var_142, var_146);
            var_148 = wp::add(var_145, var_147);
            var_149 = wp::add(var_148, var_143);
            var_150 = eval_basis_trilinear_0(var_86, var_149);
            // gi = ci + li                                                                       <L 289>
            var_151 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 290>
            var_152 = wp::add(var_48, var_142);
            // gk = ck + lk                                                                       <L 291>
            var_153 = wp::add(var_58, var_143);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_154 = wp::mul(var_151, var_88);
            var_155 = wp::mul(var_154, var_90);
            var_156 = wp::mul(var_152, var_90);
            var_157 = wp::add(var_155, var_156);
            var_158 = wp::add(var_157, var_153);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_159 = wp::add(var_92, var_158);
            var_160 = wp::address(var_flexnode_xpos_in, var_0, var_159);
            var_162 = wp::load(var_160);
            var_161 = wp::mul(var_150, var_162);
            var_163 = wp::add(var_141, var_161);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_166 = wp::mul(var_98, var_165);
            var_168 = wp::mul(var_142, var_167);
            var_169 = wp::add(var_166, var_168);
            var_170 = wp::add(var_169, var_164);
            var_171 = eval_basis_trilinear_0(var_86, var_170);
            // gi = ci + li                                                                       <L 289>
            var_172 = wp::add(var_38, var_98);
            // gj = cj + lj                                                                       <L 290>
            var_173 = wp::add(var_48, var_142);
            // gk = ck + lk                                                                       <L 291>
            var_174 = wp::add(var_58, var_164);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_175 = wp::mul(var_172, var_88);
            var_176 = wp::mul(var_175, var_90);
            var_177 = wp::mul(var_173, var_90);
            var_178 = wp::add(var_176, var_177);
            var_179 = wp::add(var_178, var_174);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_180 = wp::add(var_92, var_179);
            var_181 = wp::address(var_flexnode_xpos_in, var_0, var_180);
            var_183 = wp::load(var_181);
            var_182 = wp::mul(var_171, var_183);
            var_184 = wp::add(var_163, var_182);
            // for lj in range(2):                                                                <L 286>
            // for lk in range(2):                                                                <L 287>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_189 = wp::mul(var_185, var_188);
            var_191 = wp::mul(var_186, var_190);
            var_192 = wp::add(var_189, var_191);
            var_193 = wp::add(var_192, var_187);
            var_194 = eval_basis_trilinear_0(var_86, var_193);
            // gi = ci + li                                                                       <L 289>
            var_195 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 290>
            var_196 = wp::add(var_48, var_186);
            // gk = ck + lk                                                                       <L 291>
            var_197 = wp::add(var_58, var_187);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_198 = wp::mul(var_195, var_88);
            var_199 = wp::mul(var_198, var_90);
            var_200 = wp::mul(var_196, var_90);
            var_201 = wp::add(var_199, var_200);
            var_202 = wp::add(var_201, var_197);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_203 = wp::add(var_92, var_202);
            var_204 = wp::address(var_flexnode_xpos_in, var_0, var_203);
            var_206 = wp::load(var_204);
            var_205 = wp::mul(var_194, var_206);
            var_207 = wp::add(var_184, var_205);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_210 = wp::mul(var_185, var_209);
            var_212 = wp::mul(var_186, var_211);
            var_213 = wp::add(var_210, var_212);
            var_214 = wp::add(var_213, var_208);
            var_215 = eval_basis_trilinear_0(var_86, var_214);
            // gi = ci + li                                                                       <L 289>
            var_216 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 290>
            var_217 = wp::add(var_48, var_186);
            // gk = ck + lk                                                                       <L 291>
            var_218 = wp::add(var_58, var_208);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_219 = wp::mul(var_216, var_88);
            var_220 = wp::mul(var_219, var_90);
            var_221 = wp::mul(var_217, var_90);
            var_222 = wp::add(var_220, var_221);
            var_223 = wp::add(var_222, var_218);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_224 = wp::add(var_92, var_223);
            var_225 = wp::address(var_flexnode_xpos_in, var_0, var_224);
            var_227 = wp::load(var_225);
            var_226 = wp::mul(var_215, var_227);
            var_228 = wp::add(var_207, var_226);
            // for lk in range(2):                                                                <L 287>
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_232 = wp::mul(var_185, var_231);
            var_234 = wp::mul(var_229, var_233);
            var_235 = wp::add(var_232, var_234);
            var_236 = wp::add(var_235, var_230);
            var_237 = eval_basis_trilinear_0(var_86, var_236);
            // gi = ci + li                                                                       <L 289>
            var_238 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 290>
            var_239 = wp::add(var_48, var_229);
            // gk = ck + lk                                                                       <L 291>
            var_240 = wp::add(var_58, var_230);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_241 = wp::mul(var_238, var_88);
            var_242 = wp::mul(var_241, var_90);
            var_243 = wp::mul(var_239, var_90);
            var_244 = wp::add(var_242, var_243);
            var_245 = wp::add(var_244, var_240);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_246 = wp::add(var_92, var_245);
            var_247 = wp::address(var_flexnode_xpos_in, var_0, var_246);
            var_249 = wp::load(var_247);
            var_248 = wp::mul(var_237, var_249);
            var_250 = wp::add(var_228, var_248);
            // w = support.eval_basis_trilinear(local, li * 4 + lj * 2 + lk)                      <L 288>
            var_253 = wp::mul(var_185, var_252);
            var_255 = wp::mul(var_229, var_254);
            var_256 = wp::add(var_253, var_255);
            var_257 = wp::add(var_256, var_251);
            var_258 = eval_basis_trilinear_0(var_86, var_257);
            // gi = ci + li                                                                       <L 289>
            var_259 = wp::add(var_38, var_185);
            // gj = cj + lj                                                                       <L 290>
            var_260 = wp::add(var_48, var_229);
            // gk = ck + lk                                                                       <L 291>
            var_261 = wp::add(var_58, var_251);
            // node_idx = gi * ny_g * nz_g + gj * nz_g + gk                                       <L 292>
            var_262 = wp::mul(var_259, var_88);
            var_263 = wp::mul(var_262, var_90);
            var_264 = wp::mul(var_260, var_90);
            var_265 = wp::add(var_263, var_264);
            var_266 = wp::add(var_265, var_261);
            // result += w * flexnode_xpos_in[worldid, nstart + node_idx]                         <L 293>
            var_267 = wp::add(var_92, var_266);
            var_268 = wp::address(var_flexnode_xpos_in, var_0, var_267);
            var_270 = wp::load(var_268);
            var_269 = wp::mul(var_258, var_270);
            var_271 = wp::add(var_250, var_269);
            // flexvert_xpos_out[worldid, vertid] = result                                        <L 295>
            wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_271);
        }
        if (!var_15) {
            // bodyid = flex_vertbodyid[vertid]                                                   <L 298>
            var_272 = wp::address(var_flex_vertbodyid, var_1);
            var_274 = wp::load(var_272);
            var_273 = wp::copy(var_274);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 299>
            var_275 = wp::address(var_xpos_in, var_0, var_273);
            var_277 = wp::load(var_275);
            var_276 = wp::copy(var_277);
            // if flex_centered[f]:                                                               <L 301>
            var_278 = wp::address(var_flex_centered, var_3);
            var_279 = wp::load(var_278);
            if (var_279) {
                // flexvert_xpos_out[worldid, vertid] = xpos                                      <L 302>
                wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_276);
            }
            var_280 = wp::load(var_278);
            var_281 = wp::load(var_278);
            if (!var_281) {
                // xmat = xmat_in[worldid, bodyid]                                                <L 304>
                var_282 = wp::address(var_xmat_in, var_0, var_273);
                var_284 = wp::load(var_282);
                var_283 = wp::copy(var_284);
                // local_pos = flex_vert[vertid]                                                  <L 305>
                var_285 = wp::address(var_flex_vert, var_1);
                var_287 = wp::load(var_285);
                var_286 = wp::copy(var_287);
                // flexvert_xpos_out[worldid, vertid] = xmat @ local_pos + xpos                   <L 306>
                var_288 = wp::mul(var_283, var_286);
                var_289 = wp::add(var_288, var_276);
                wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_289);
            }
            var_290 = wp::load(var_278);
        }
    }
}



extern "C" __global__ void _cacc_world_6ec26fdb_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::vec_t<3, wp::float32> var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        //---------
        // forward
        // def _cacc_world(                                                                       <L 1354>
        // worldid = wp.tid()                                                                     <L 1360>
        var_0 = builtin_tid1d();
        // cacc_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -gravity[worldid % gravity.shape[0]])       <L 1361>
        var_2 = wp::vec_t<3, wp::float32>(var_1);
        var_3 = &(var_gravity.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_gravity, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::neg(var_10);
        var_11 = wp::vec_t<6, wp::float32>(var_2, var_9);
        wp::array_store(var_cacc_out, var_0, var_12, var_11);
    }
}



extern "C" __global__ void _cfrc_backward_72237d38_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::array_t<wp::vec_t<6, wp::float32>> var_10;
        wp::vec_t<6, wp::float32>* var_11;
        wp::vec_t<6, wp::float32> var_12;
        wp::vec_t<6, wp::float32> var_13;
        //---------
        // forward
        // def _cfrc_backward(                                                                    <L 1460>
        // worldid, nodeid = wp.tid()                                                             <L 1470>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 1471>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 1472>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if bodyid != 0:                                                                        <L 1473>
        var_9 = (var_3 != var_8);
        if (var_9) {
            // wp.atomic_add(cfrc_int_out[worldid], pid, cfrc_int_in[worldid, bodyid])            <L 1474>
            var_10 = wp::view(var_cfrc_int_out, var_0);
            var_11 = wp::address(var_cfrc_int_in, var_0, var_3);
            var_13 = wp::load(var_11);
            var_12 = wp::atomic_add(var_10, var_6, var_13);
        }
    }
}



extern "C" __global__ void _flex_nodes_7c3ae771_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_flex_nodeadr,
    wp::array_t<wp::int32> var_flex_nodenum,
    wp::array_t<wp::int32> var_flex_nodebodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_node,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        bool* var_19;
        bool var_20;
        bool var_21;
        bool var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        bool var_26;
        const wp::int32 var_27 = 0;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.0;
        bool var_30;
        const wp::int32 var_31 = 1;
        wp::float32 var_32;
        const wp::float32 var_33 = 0.0;
        bool var_34;
        const wp::int32 var_35 = 2;
        wp::float32 var_36;
        const wp::float32 var_37 = 0.0;
        bool var_38;
        wp::mat_t<3, 3, wp::float32>* var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        wp::mat_t<3, 3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        bool var_44;
        //---------
        // forward
        // def _flex_nodes(                                                                       <L 310>
        // worldid, nodeid = wp.tid()                                                             <L 324>
        builtin_tid2d(var_0, var_1);
        // for f in range(nflex):                                                                 <L 326>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = nodeid - flex_nodeadr[f]                                                   <L 327>
            var_4 = wp::address(var_flex_nodeadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_nodenum[f]:                                         <L 328>
            var_9 = (var_5 >= var_8);
            var_7 = var_9;
            if (var_7) {
                var_10 = wp::address(var_flex_nodenum, var_3);
                var_12 = wp::load(var_10);
                var_11 = (var_5 < var_12);
                var_7 = var_7 && var_11;
            }
            if (var_7) {
                // break                                                                          <L 329>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // bodyid = flex_nodebodyid[nodeid]                                                       <L 331>
        var_13 = wp::address(var_flex_nodebodyid, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 332>
        var_16 = wp::address(var_xpos_in, var_0, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if flex_centered[f]:                                                                   <L 334>
        var_19 = wp::address(var_flex_centered, var_3);
        var_20 = wp::load(var_19);
        if (var_20) {
            // flexnode_xpos_out[worldid, nodeid] = xpos                                          <L 335>
            wp::array_store(var_flexnode_xpos_out, var_0, var_1, var_17);
        }
        var_21 = wp::load(var_19);
        var_22 = wp::load(var_19);
        if (!var_22) {
            // local_pos = flex_node[nodeid]                                                      <L 337>
            var_23 = wp::address(var_flex_node, var_1);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if local_pos[0] == 0.0 and local_pos[1] == 0.0 and local_pos[2] == 0.0:            <L 338>
            var_28 = wp::extract(var_24, var_27);
            var_30 = (var_28 == var_29);
            var_26 = var_30;
            if (var_26) {
                var_32 = wp::extract(var_24, var_31);
                var_34 = (var_32 == var_33);
                var_26 = var_26 && var_34;
            }
            if (var_26) {
                var_36 = wp::extract(var_24, var_35);
                var_38 = (var_36 == var_37);
                var_26 = var_26 && var_38;
            }
            if (var_26) {
                // flexnode_xpos_out[worldid, nodeid] = xpos                                      <L 339>
                wp::array_store(var_flexnode_xpos_out, var_0, var_1, var_17);
            }
            if (!var_26) {
                // xmat = xmat_in[worldid, bodyid]                                                <L 341>
                var_39 = wp::address(var_xmat_in, var_0, var_14);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // flexnode_xpos_out[worldid, nodeid] = xmat @ local_pos + xpos                   <L 342>
                var_42 = wp::mul(var_40, var_24);
                var_43 = wp::add(var_42, var_17);
                wp::array_store(var_flexnode_xpos_out, var_0, var_1, var_43);
            }
        }
        var_44 = wp::load(var_19);
    }
}



extern "C" __global__ void _linear_momentum_813ce6af_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::array_t<wp::vec_t<3, wp::float32>> var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        const wp::float32 var_12 = 1e-15;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        //---------
        // forward
        // def _linear_momentum(                                                                  <L 3539>
        // worldid, nodeid = wp.tid()                                                             <L 3550>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 3551>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if bodyid:                                                                             <L 3552>
        if (var_3) {
            // pid = body_parentid[bodyid]                                                        <L 3553>
            var_5 = wp::address(var_body_parentid, var_3);
            var_7 = wp::load(var_5);
            var_6 = wp::copy(var_7);
            // wp.atomic_add(subtree_linvel_out[worldid], pid, subtree_linvel_in[worldid, bodyid])       <L 3554>
            var_8 = wp::view(var_subtree_linvel_out, var_0);
            var_9 = wp::address(var_subtree_linvel_in, var_0, var_3);
            var_11 = wp::load(var_9);
            var_10 = wp::atomic_add(var_8, var_6, var_11);
        }
        // subtree_linvel_out[worldid, bodyid] /= wp.max(MJ_MINVAL, body_subtreemass[worldid % body_subtreemass.shape[0], bodyid])       <L 3555>
        var_13 = &(var_body_subtreemass.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_body_subtreemass, var_17, var_3);
        var_20 = wp::load(var_18);
        var_19 = wp::max(var_12, var_20);
        var_21 = wp::address(var_subtree_linvel_out, var_0, var_3);
        var_23 = wp::load(var_21);
        var_22 = wp::div(var_23, var_19);
        wp::array_store(var_subtree_linvel_out, var_0, var_3, var_22);
    }
}



extern "C" __global__ void _kinematics_branch_95f8028e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_body_quat,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_jnt_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_jnt_axis,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mocap_pos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_mocap_quat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_out,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xanchor_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xaxis_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::array_t<wp::float32> var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        bool var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::float32* var_38;
        const wp::int32 var_39 = 2;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 3;
        wp::int32 var_47;
        wp::float32* var_48;
        const wp::int32 var_49 = 4;
        wp::int32 var_50;
        wp::float32* var_51;
        const wp::int32 var_52 = 5;
        wp::int32 var_53;
        wp::float32* var_54;
        const wp::int32 var_55 = 6;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::quat_t<wp::float32> var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::quat_t<wp::float32> var_63;
        wp::shape_t* var_64;
        const wp::int32 var_65 = 0;
        wp::int32 var_66;
        wp::shape_t var_67;
        wp::int32 var_68;
        wp::vec_t<3, wp::float32>* var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::shape_t* var_71;
        const wp::int32 var_72 = 0;
        wp::int32 var_73;
        wp::shape_t var_74;
        wp::int32 var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::int32* var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        const wp::int32 var_82 = 0;
        bool var_83;
        wp::vec_t<3, wp::float32>* var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::quat_t<wp::float32>* var_87;
        wp::quat_t<wp::float32> var_88;
        wp::quat_t<wp::float32> var_89;
        wp::shape_t* var_90;
        const wp::int32 var_91 = 0;
        wp::int32 var_92;
        wp::shape_t var_93;
        wp::int32 var_94;
        wp::vec_t<3, wp::float32>* var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::shape_t* var_98;
        const wp::int32 var_99 = 0;
        wp::int32 var_100;
        wp::shape_t var_101;
        wp::int32 var_102;
        wp::quat_t<wp::float32>* var_103;
        wp::quat_t<wp::float32> var_104;
        wp::quat_t<wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::quat_t<wp::float32> var_107;
        const wp::int32 var_108 = 0;
        bool var_109;
        wp::quat_t<wp::float32>* var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::quat_t<wp::float32> var_112;
        wp::vec_t<3, wp::float32>* var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::quat_t<wp::float32>* var_116;
        wp::quat_t<wp::float32> var_117;
        wp::quat_t<wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::quat_t<wp::float32> var_120;
        wp::range_t var_121;
        wp::int32 var_122;
        wp::int32* var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32* var_126;
        wp::int32 var_127;
        wp::int32 var_128;
        wp::shape_t* var_129;
        const wp::int32 var_130 = 0;
        wp::int32 var_131;
        wp::shape_t var_132;
        wp::int32 var_133;
        wp::vec_t<3, wp::float32>* var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32>* var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        const wp::int32 var_142 = 1;
        bool var_143;
        const wp::int32 var_144 = 0;
        wp::int32 var_145;
        wp::float32* var_146;
        const wp::int32 var_147 = 1;
        wp::int32 var_148;
        wp::float32* var_149;
        const wp::int32 var_150 = 2;
        wp::int32 var_151;
        wp::float32* var_152;
        const wp::int32 var_153 = 3;
        wp::int32 var_154;
        wp::float32* var_155;
        wp::quat_t<wp::float32> var_156;
        wp::float32 var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::quat_t<wp::float32> var_161;
        wp::quat_t<wp::float32> var_162;
        wp::vec_t<3, wp::float32>* var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        const wp::int32 var_167 = 2;
        bool var_168;
        wp::float32* var_169;
        wp::shape_t* var_170;
        const wp::int32 var_171 = 0;
        wp::int32 var_172;
        wp::shape_t var_173;
        wp::int32 var_174;
        wp::float32* var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::vec_t<3, wp::float32> var_179;
        wp::vec_t<3, wp::float32> var_180;
        const wp::int32 var_181 = 3;
        bool var_182;
        wp::shape_t* var_183;
        const wp::int32 var_184 = 0;
        wp::int32 var_185;
        wp::shape_t var_186;
        wp::int32 var_187;
        wp::float32* var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::float32* var_191;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::quat_t<wp::float32> var_194;
        wp::quat_t<wp::float32> var_195;
        wp::vec_t<3, wp::float32>* var_196;
        wp::vec_t<3, wp::float32> var_197;
        wp::vec_t<3, wp::float32> var_198;
        wp::vec_t<3, wp::float32> var_199;
        wp::vec_t<3, wp::float32> var_200;
        wp::quat_t<wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        wp::quat_t<wp::float32> var_203;
        wp::vec_t<3, wp::float32> var_204;
        wp::quat_t<wp::float32> var_205;
        const wp::int32 var_206 = 1;
        wp::int32 var_207;
        wp::quat_t<wp::float32> var_208;
        //---------
        // forward
        // def _kinematics_branch(                                                                <L 47>
        // worldid, branchid = wp.tid()                                                           <L 72>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 74>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 75>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qpos = qpos_in[worldid]                                                                <L 77>
        var_10 = wp::view(var_qpos_in, var_0);
        // for i in range(start, end):                                                            <L 79>
        var_11 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
            // bodyid = body_branches[i]                                                          <L 80>
            var_13 = wp::address(var_body_branches, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // pid = body_parentid[bodyid]                                                        <L 81>
            var_16 = wp::address(var_body_parentid, var_14);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // jntadr = body_jntadr[bodyid]                                                       <L 82>
            var_19 = wp::address(var_body_jntadr, var_14);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // jntnum = body_jntnum[bodyid]                                                       <L 83>
            var_22 = wp::address(var_body_jntnum, var_14);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // if jntnum == 1:                                                                    <L 85>
            var_26 = (var_23 == var_25);
            if (var_26) {
                // jnt_type_ = jnt_type[jntadr]                                                   <L 86>
                var_27 = wp::address(var_jnt_type, var_20);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // if jnt_type_ == JointType.FREE:                                                <L 87>
                var_31 = (var_28 == var_30);
                if (var_31) {
                    // qadr = jnt_qposadr[jntadr]                                                 <L 88>
                    var_32 = wp::address(var_jnt_qposadr, var_20);
                    var_34 = wp::load(var_32);
                    var_33 = wp::copy(var_34);
                    // xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])                 <L 89>
                    var_35 = wp::address(var_10, var_33);
                    var_37 = wp::add(var_33, var_36);
                    var_38 = wp::address(var_10, var_37);
                    var_40 = wp::add(var_33, var_39);
                    var_41 = wp::address(var_10, var_40);
                    var_43 = wp::load(var_35);
                    var_44 = wp::load(var_38);
                    var_45 = wp::load(var_41);
                    var_42 = wp::vec_t<3, wp::float32>(var_43, var_44, var_45);
                    // xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])       <L 90>
                    var_47 = wp::add(var_33, var_46);
                    var_48 = wp::address(var_10, var_47);
                    var_50 = wp::add(var_33, var_49);
                    var_51 = wp::address(var_10, var_50);
                    var_53 = wp::add(var_33, var_52);
                    var_54 = wp::address(var_10, var_53);
                    var_56 = wp::add(var_33, var_55);
                    var_57 = wp::address(var_10, var_56);
                    var_59 = wp::load(var_48);
                    var_60 = wp::load(var_51);
                    var_61 = wp::load(var_54);
                    var_62 = wp::load(var_57);
                    var_58 = wp::quat_t<wp::float32>(var_59, var_60, var_61, var_62);
                    // xquat = wp.normalize(xquat)                                                <L 91>
                    var_63 = wp::normalize(var_58);
                    // xpos_out[worldid, bodyid] = xpos                                           <L 93>
                    wp::array_store(var_xpos_out, var_0, var_14, var_42);
                    // xquat_out[worldid, bodyid] = xquat                                         <L 94>
                    wp::array_store(var_xquat_out, var_0, var_14, var_63);
                    // xanchor_out[worldid, jntadr] = xpos                                        <L 95>
                    wp::array_store(var_xanchor_out, var_0, var_20, var_42);
                    // xaxis_out[worldid, jntadr] = jnt_axis[worldid % jnt_axis.shape[0], jntadr]       <L 96>
                    var_64 = &(var_jnt_axis.shape);
                    var_67 = wp::load(var_64);
                    var_66 = wp::extract(var_67, var_65);
                    var_68 = wp::mod(var_0, var_66);
                    var_69 = wp::address(var_jnt_axis, var_68, var_20);
                    var_70 = wp::load(var_69);
                    wp::array_store(var_xaxis_out, var_0, var_20, var_70);
                    // continue                                                                   <L 97>
                    goto start_for_0;
                }
            }
            // jnt_pos_id = worldid % jnt_pos.shape[0]                                            <L 101>
            var_71 = &(var_jnt_pos.shape);
            var_74 = wp::load(var_71);
            var_73 = wp::extract(var_74, var_72);
            var_75 = wp::mod(var_0, var_73);
            // pid = body_parentid[bodyid]                                                        <L 102>
            var_76 = wp::address(var_body_parentid, var_14);
            var_78 = wp::load(var_76);
            var_77 = wp::copy(var_78);
            // mocapid = body_mocapid[bodyid]                                                     <L 105>
            var_79 = wp::address(var_body_mocapid, var_14);
            var_81 = wp::load(var_79);
            var_80 = wp::copy(var_81);
            // if mocapid >= 0:                                                                   <L 106>
            var_83 = (var_80 >= var_82);
            if (var_83) {
                // xpos = mocap_pos_in[worldid, mocapid]                                          <L 107>
                var_84 = wp::address(var_mocap_pos_in, var_0, var_80);
                var_86 = wp::load(var_84);
                var_85 = wp::copy(var_86);
                // xquat = mocap_quat_in[worldid, mocapid]                                        <L 108>
                var_87 = wp::address(var_mocap_quat_in, var_0, var_80);
                var_89 = wp::load(var_87);
                var_88 = wp::copy(var_89);
            }
            if (!var_83) {
                // xpos = body_pos[worldid % body_pos.shape[0], bodyid]                           <L 110>
                var_90 = &(var_body_pos.shape);
                var_93 = wp::load(var_90);
                var_92 = wp::extract(var_93, var_91);
                var_94 = wp::mod(var_0, var_92);
                var_95 = wp::address(var_body_pos, var_94, var_14);
                var_97 = wp::load(var_95);
                var_96 = wp::copy(var_97);
                // xquat = body_quat[worldid % body_quat.shape[0], bodyid]                        <L 111>
                var_98 = &(var_body_quat.shape);
                var_101 = wp::load(var_98);
                var_100 = wp::extract(var_101, var_99);
                var_102 = wp::mod(var_0, var_100);
                var_103 = wp::address(var_body_quat, var_102, var_14);
                var_105 = wp::load(var_103);
                var_104 = wp::copy(var_105);
            }
            var_106 = wp::where(var_83, var_85, var_96);
            var_107 = wp::where(var_83, var_88, var_104);
            // if pid >= 0:                                                                       <L 113>
            var_109 = (var_77 >= var_108);
            if (var_109) {
                // xpos = math.rot_vec_quat(xpos, xquat_out[worldid, pid]) + xpos_out[worldid, pid]       <L 114>
                var_110 = wp::address(var_xquat_out, var_0, var_77);
                var_112 = wp::load(var_110);
                var_111 = rot_vec_quat_0(var_106, var_112);
                var_113 = wp::address(var_xpos_out, var_0, var_77);
                var_115 = wp::load(var_113);
                var_114 = wp::add(var_111, var_115);
                // xquat = math.mul_quat(xquat_out[worldid, pid], xquat)                          <L 115>
                var_116 = wp::address(var_xquat_out, var_0, var_77);
                var_118 = wp::load(var_116);
                var_117 = mul_quat_0(var_118, var_107);
            }
            var_119 = wp::where(var_109, var_114, var_106);
            var_120 = wp::where(var_109, var_117, var_107);
            // for _ in range(jntnum):                                                            <L 117>
            var_121 = wp::range(var_23);
            start_for_2:;
                if (iter_cmp(var_121) == 0) goto end_for_2;
                var_122 = wp::iter_next(var_121);
                // qadr = jnt_qposadr[jntadr]                                                     <L 118>
                var_123 = wp::address(var_jnt_qposadr, var_20);
                var_125 = wp::load(var_123);
                var_124 = wp::copy(var_125);
                // jnt_type_ = jnt_type[jntadr]                                                   <L 119>
                var_126 = wp::address(var_jnt_type, var_20);
                var_128 = wp::load(var_126);
                var_127 = wp::copy(var_128);
                // jnt_axis_ = jnt_axis[worldid % jnt_axis.shape[0], jntadr]                      <L 120>
                var_129 = &(var_jnt_axis.shape);
                var_132 = wp::load(var_129);
                var_131 = wp::extract(var_132, var_130);
                var_133 = wp::mod(var_0, var_131);
                var_134 = wp::address(var_jnt_axis, var_133, var_20);
                var_136 = wp::load(var_134);
                var_135 = wp::copy(var_136);
                // xanchor = math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat) + xpos         <L 121>
                var_137 = wp::address(var_jnt_pos, var_75, var_20);
                var_139 = wp::load(var_137);
                var_138 = rot_vec_quat_0(var_139, var_120);
                var_140 = wp::add(var_138, var_119);
                // xaxis = math.rot_vec_quat(jnt_axis_, xquat)                                    <L 122>
                var_141 = rot_vec_quat_0(var_135, var_120);
                // if jnt_type_ == JointType.BALL:                                                <L 124>
                var_143 = (var_127 == var_142);
                if (var_143) {
                    // qloc = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])       <L 125>
                    var_145 = wp::add(var_124, var_144);
                    var_146 = wp::address(var_10, var_145);
                    var_148 = wp::add(var_124, var_147);
                    var_149 = wp::address(var_10, var_148);
                    var_151 = wp::add(var_124, var_150);
                    var_152 = wp::address(var_10, var_151);
                    var_154 = wp::add(var_124, var_153);
                    var_155 = wp::address(var_10, var_154);
                    var_157 = wp::load(var_146);
                    var_158 = wp::load(var_149);
                    var_159 = wp::load(var_152);
                    var_160 = wp::load(var_155);
                    var_156 = wp::quat_t<wp::float32>(var_157, var_158, var_159, var_160);
                    // qloc = wp.normalize(qloc)                                                  <L 126>
                    var_161 = wp::normalize(var_156);
                    // xquat = math.mul_quat(xquat, qloc)                                         <L 127>
                    var_162 = mul_quat_0(var_120, var_161);
                    // xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)       <L 129>
                    var_163 = wp::address(var_jnt_pos, var_75, var_20);
                    var_165 = wp::load(var_163);
                    var_164 = rot_vec_quat_0(var_165, var_162);
                    var_166 = wp::sub(var_140, var_164);
                }
                if (!var_143) {
                    // elif jnt_type_ == JointType.SLIDE:                                         <L 130>
                    var_168 = (var_127 == var_167);
                    if (var_168) {
                        // xpos += xaxis * (qpos[qadr] - qpos0[worldid % qpos0.shape[0], qadr])       <L 131>
                        var_169 = wp::address(var_10, var_124);
                        var_170 = &(var_qpos0.shape);
                        var_173 = wp::load(var_170);
                        var_172 = wp::extract(var_173, var_171);
                        var_174 = wp::mod(var_0, var_172);
                        var_175 = wp::address(var_qpos0, var_174, var_124);
                        var_177 = wp::load(var_169);
                        var_178 = wp::load(var_175);
                        var_176 = wp::sub(var_177, var_178);
                        var_179 = wp::mul(var_141, var_176);
                        var_180 = wp::add(var_119, var_179);
                    }
                    if (!var_168) {
                        // elif jnt_type_ == JointType.HINGE:                                     <L 132>
                        var_182 = (var_127 == var_181);
                        if (var_182) {
                            // qpos0_ = qpos0[worldid % qpos0.shape[0], qadr]                     <L 133>
                            var_183 = &(var_qpos0.shape);
                            var_186 = wp::load(var_183);
                            var_185 = wp::extract(var_186, var_184);
                            var_187 = wp::mod(var_0, var_185);
                            var_188 = wp::address(var_qpos0, var_187, var_124);
                            var_190 = wp::load(var_188);
                            var_189 = wp::copy(var_190);
                            // qloc_ = math.axis_angle_to_quat(jnt_axis_, qpos[qadr] - qpos0_)       <L 134>
                            var_191 = wp::address(var_10, var_124);
                            var_193 = wp::load(var_191);
                            var_192 = wp::sub(var_193, var_189);
                            var_194 = axis_angle_to_quat_0(var_135, var_192);
                            // xquat = math.mul_quat(xquat, qloc_)                                <L 135>
                            var_195 = mul_quat_0(var_120, var_194);
                            // xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)       <L 137>
                            var_196 = wp::address(var_jnt_pos, var_75, var_20);
                            var_198 = wp::load(var_196);
                            var_197 = rot_vec_quat_0(var_198, var_195);
                            var_199 = wp::sub(var_140, var_197);
                        }
                        var_200 = wp::where(var_182, var_199, var_119);
                        var_201 = wp::where(var_182, var_195, var_120);
                    }
                    var_202 = wp::where(var_168, var_180, var_200);
                    var_203 = wp::where(var_168, var_120, var_201);
                }
                var_204 = wp::where(var_143, var_166, var_202);
                var_205 = wp::where(var_143, var_162, var_203);
                // xanchor_out[worldid, jntadr] = xanchor                                         <L 139>
                wp::array_store(var_xanchor_out, var_0, var_20, var_140);
                // xaxis_out[worldid, jntadr] = xaxis                                             <L 140>
                wp::array_store(var_xaxis_out, var_0, var_20, var_141);
                // jntadr += 1                                                                    <L 141>
                var_207 = wp::add(var_20, var_206);
                wp::assign(var_20, var_207);
                wp::assign(var_28, var_127);
                wp::assign(var_33, var_124);
                wp::assign(var_119, var_204);
                wp::assign(var_120, var_205);
                goto start_for_2;
            end_for_2:;
            // xquat = wp.normalize(xquat)                                                        <L 143>
            var_208 = wp::normalize(var_120);
            // xpos_out[worldid, bodyid] = xpos                                                   <L 144>
            wp::array_store(var_xpos_out, var_0, var_14, var_119);
            // xquat_out[worldid, bodyid] = xquat                                                 <L 145>
            wp::array_store(var_xquat_out, var_0, var_14, var_208);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _flex_face_kinematics_9a1620ae_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::vec_t<3, wp::int32>> var_flex_cellnum,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_face_map,
    wp::array_t<wp::int32> var_flex_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_xpos_out,
    wp::array_t<wp::quat_t<wp::float32>> var_face_quat_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::int32>* var_13;
        wp::vec_t<3, wp::int32> var_14;
        wp::vec_t<3, wp::int32> var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        const wp::int32 var_20 = 2;
        wp::int32 var_21;
        const wp::float32 var_22 = 0.0;
        wp::vec_t<3, wp::float32> var_23;
        const wp::float32 var_24 = 0.0;
        wp::vec_t<3, wp::float32> var_25;
        const wp::int32 var_26 = 0;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = -1;
        bool var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 2;
        bool var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::float32 var_45;
        const wp::float32 var_46 = -1.0;
        const wp::float32 var_47 = 2.0;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 2;
        bool var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::float32 var_56;
        const wp::float32 var_57 = -1.0;
        const wp::float32 var_58 = 2.0;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 2;
        bool var_64;
        const wp::int32 var_65 = 1;
        bool var_66;
        const wp::float32 var_67 = 1.0;
        const wp::float32 var_68 = 0.0;
        wp::float32 var_69;
        const wp::float32 var_70 = 0.5;
        wp::float32 var_71;
        const wp::int32 var_72 = 2;
        bool var_73;
        const wp::int32 var_74 = 1;
        bool var_75;
        const wp::float32 var_76 = 1.0;
        const wp::float32 var_77 = 0.0;
        wp::float32 var_78;
        const wp::float32 var_79 = 0.5;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        const wp::float32 var_87 = 0.0;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        const wp::int32 var_91 = 1;
        wp::int32* var_92;
        wp::int32 var_93;
        wp::int32 var_94;
        const wp::int32 var_95 = -1;
        bool var_96;
        wp::vec_t<3, wp::float32>* var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        const wp::int32 var_100 = 1;
        wp::int32 var_101;
        wp::int32 var_102;
        const wp::int32 var_103 = 1;
        wp::int32 var_104;
        wp::int32 var_105;
        const wp::int32 var_106 = 2;
        bool var_107;
        const wp::int32 var_108 = 1;
        wp::int32 var_109;
        wp::float32 var_110;
        const wp::float32 var_111 = -1.0;
        const wp::float32 var_112 = 2.0;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 2;
        bool var_118;
        const wp::int32 var_119 = 1;
        wp::int32 var_120;
        wp::float32 var_121;
        const wp::float32 var_122 = -1.0;
        const wp::float32 var_123 = 2.0;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 2;
        bool var_129;
        const wp::int32 var_130 = 1;
        bool var_131;
        const wp::float32 var_132 = 1.0;
        const wp::float32 var_133 = 0.0;
        wp::float32 var_134;
        const wp::float32 var_135 = 0.5;
        wp::float32 var_136;
        const wp::int32 var_137 = 2;
        bool var_138;
        const wp::int32 var_139 = 1;
        bool var_140;
        const wp::float32 var_141 = 1.0;
        const wp::float32 var_142 = 0.0;
        wp::float32 var_143;
        const wp::float32 var_144 = 0.5;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        wp::vec_t<3, wp::float32> var_151;
        const wp::float32 var_152 = 0.0;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        const wp::int32 var_165 = 2;
        wp::int32* var_166;
        wp::int32 var_167;
        wp::int32 var_168;
        const wp::int32 var_169 = -1;
        bool var_170;
        wp::vec_t<3, wp::float32>* var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        const wp::int32 var_174 = 1;
        wp::int32 var_175;
        wp::int32 var_176;
        const wp::int32 var_177 = 1;
        wp::int32 var_178;
        wp::int32 var_179;
        const wp::int32 var_180 = 2;
        bool var_181;
        const wp::int32 var_182 = 1;
        wp::int32 var_183;
        wp::float32 var_184;
        const wp::float32 var_185 = -1.0;
        const wp::float32 var_186 = 2.0;
        wp::float32 var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        const wp::int32 var_191 = 2;
        bool var_192;
        const wp::int32 var_193 = 1;
        wp::int32 var_194;
        wp::float32 var_195;
        const wp::float32 var_196 = -1.0;
        const wp::float32 var_197 = 2.0;
        wp::float32 var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        const wp::int32 var_202 = 2;
        bool var_203;
        const wp::int32 var_204 = 1;
        bool var_205;
        const wp::float32 var_206 = 1.0;
        const wp::float32 var_207 = 0.0;
        wp::float32 var_208;
        const wp::float32 var_209 = 0.5;
        wp::float32 var_210;
        const wp::int32 var_211 = 2;
        bool var_212;
        const wp::int32 var_213 = 1;
        bool var_214;
        const wp::float32 var_215 = 1.0;
        const wp::float32 var_216 = 0.0;
        wp::float32 var_217;
        const wp::float32 var_218 = 0.5;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::vec_t<3, wp::float32> var_222;
        wp::vec_t<3, wp::float32> var_223;
        wp::vec_t<3, wp::float32> var_224;
        wp::vec_t<3, wp::float32> var_225;
        const wp::float32 var_226 = 0.0;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::float32 var_238;
        const wp::int32 var_239 = 3;
        wp::int32* var_240;
        wp::int32 var_241;
        wp::int32 var_242;
        const wp::int32 var_243 = -1;
        bool var_244;
        wp::vec_t<3, wp::float32>* var_245;
        wp::vec_t<3, wp::float32> var_246;
        wp::vec_t<3, wp::float32> var_247;
        const wp::int32 var_248 = 1;
        wp::int32 var_249;
        wp::int32 var_250;
        const wp::int32 var_251 = 1;
        wp::int32 var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 2;
        bool var_255;
        const wp::int32 var_256 = 1;
        wp::int32 var_257;
        wp::float32 var_258;
        const wp::float32 var_259 = -1.0;
        const wp::float32 var_260 = 2.0;
        wp::float32 var_261;
        wp::float32 var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        const wp::int32 var_265 = 2;
        bool var_266;
        const wp::int32 var_267 = 1;
        wp::int32 var_268;
        wp::float32 var_269;
        const wp::float32 var_270 = -1.0;
        const wp::float32 var_271 = 2.0;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        const wp::int32 var_276 = 2;
        bool var_277;
        const wp::int32 var_278 = 1;
        bool var_279;
        const wp::float32 var_280 = 1.0;
        const wp::float32 var_281 = 0.0;
        wp::float32 var_282;
        const wp::float32 var_283 = 0.5;
        wp::float32 var_284;
        const wp::int32 var_285 = 2;
        bool var_286;
        const wp::int32 var_287 = 1;
        bool var_288;
        const wp::float32 var_289 = 1.0;
        const wp::float32 var_290 = 0.0;
        wp::float32 var_291;
        const wp::float32 var_292 = 0.5;
        wp::float32 var_293;
        wp::float32 var_294;
        wp::float32 var_295;
        wp::vec_t<3, wp::float32> var_296;
        wp::vec_t<3, wp::float32> var_297;
        wp::vec_t<3, wp::float32> var_298;
        wp::vec_t<3, wp::float32> var_299;
        const wp::float32 var_300 = 0.0;
        wp::vec_t<3, wp::float32> var_301;
        wp::vec_t<3, wp::float32> var_302;
        wp::vec_t<3, wp::float32> var_303;
        wp::vec_t<3, wp::float32> var_304;
        wp::int32 var_305;
        wp::int32 var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        const wp::int32 var_313 = 4;
        wp::int32* var_314;
        wp::int32 var_315;
        wp::int32 var_316;
        const wp::int32 var_317 = -1;
        bool var_318;
        wp::vec_t<3, wp::float32>* var_319;
        wp::vec_t<3, wp::float32> var_320;
        wp::vec_t<3, wp::float32> var_321;
        const wp::int32 var_322 = 1;
        wp::int32 var_323;
        wp::int32 var_324;
        const wp::int32 var_325 = 1;
        wp::int32 var_326;
        wp::int32 var_327;
        const wp::int32 var_328 = 2;
        bool var_329;
        const wp::int32 var_330 = 1;
        wp::int32 var_331;
        wp::float32 var_332;
        const wp::float32 var_333 = -1.0;
        const wp::float32 var_334 = 2.0;
        wp::float32 var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        const wp::int32 var_339 = 2;
        bool var_340;
        const wp::int32 var_341 = 1;
        wp::int32 var_342;
        wp::float32 var_343;
        const wp::float32 var_344 = -1.0;
        const wp::float32 var_345 = 2.0;
        wp::float32 var_346;
        wp::float32 var_347;
        wp::float32 var_348;
        wp::float32 var_349;
        const wp::int32 var_350 = 2;
        bool var_351;
        const wp::int32 var_352 = 1;
        bool var_353;
        const wp::float32 var_354 = 1.0;
        const wp::float32 var_355 = 0.0;
        wp::float32 var_356;
        const wp::float32 var_357 = 0.5;
        wp::float32 var_358;
        const wp::int32 var_359 = 2;
        bool var_360;
        const wp::int32 var_361 = 1;
        bool var_362;
        const wp::float32 var_363 = 1.0;
        const wp::float32 var_364 = 0.0;
        wp::float32 var_365;
        const wp::float32 var_366 = 0.5;
        wp::float32 var_367;
        wp::float32 var_368;
        wp::float32 var_369;
        wp::vec_t<3, wp::float32> var_370;
        wp::vec_t<3, wp::float32> var_371;
        wp::vec_t<3, wp::float32> var_372;
        wp::vec_t<3, wp::float32> var_373;
        const wp::float32 var_374 = 0.0;
        wp::vec_t<3, wp::float32> var_375;
        wp::vec_t<3, wp::float32> var_376;
        wp::vec_t<3, wp::float32> var_377;
        wp::vec_t<3, wp::float32> var_378;
        wp::int32 var_379;
        wp::int32 var_380;
        wp::float32 var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        const wp::int32 var_387 = 5;
        wp::int32* var_388;
        wp::int32 var_389;
        wp::int32 var_390;
        const wp::int32 var_391 = -1;
        bool var_392;
        wp::vec_t<3, wp::float32>* var_393;
        wp::vec_t<3, wp::float32> var_394;
        wp::vec_t<3, wp::float32> var_395;
        const wp::int32 var_396 = 1;
        wp::int32 var_397;
        wp::int32 var_398;
        const wp::int32 var_399 = 1;
        wp::int32 var_400;
        wp::int32 var_401;
        const wp::int32 var_402 = 2;
        bool var_403;
        const wp::int32 var_404 = 1;
        wp::int32 var_405;
        wp::float32 var_406;
        const wp::float32 var_407 = -1.0;
        const wp::float32 var_408 = 2.0;
        wp::float32 var_409;
        wp::float32 var_410;
        wp::float32 var_411;
        wp::float32 var_412;
        const wp::int32 var_413 = 2;
        bool var_414;
        const wp::int32 var_415 = 1;
        wp::int32 var_416;
        wp::float32 var_417;
        const wp::float32 var_418 = -1.0;
        const wp::float32 var_419 = 2.0;
        wp::float32 var_420;
        wp::float32 var_421;
        wp::float32 var_422;
        wp::float32 var_423;
        const wp::int32 var_424 = 2;
        bool var_425;
        const wp::int32 var_426 = 1;
        bool var_427;
        const wp::float32 var_428 = 1.0;
        const wp::float32 var_429 = 0.0;
        wp::float32 var_430;
        const wp::float32 var_431 = 0.5;
        wp::float32 var_432;
        const wp::int32 var_433 = 2;
        bool var_434;
        const wp::int32 var_435 = 1;
        bool var_436;
        const wp::float32 var_437 = 1.0;
        const wp::float32 var_438 = 0.0;
        wp::float32 var_439;
        const wp::float32 var_440 = 0.5;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::float32 var_443;
        wp::vec_t<3, wp::float32> var_444;
        wp::vec_t<3, wp::float32> var_445;
        wp::vec_t<3, wp::float32> var_446;
        wp::vec_t<3, wp::float32> var_447;
        const wp::float32 var_448 = 0.0;
        wp::vec_t<3, wp::float32> var_449;
        wp::vec_t<3, wp::float32> var_450;
        wp::vec_t<3, wp::float32> var_451;
        wp::vec_t<3, wp::float32> var_452;
        wp::int32 var_453;
        wp::int32 var_454;
        wp::float32 var_455;
        wp::float32 var_456;
        wp::float32 var_457;
        wp::float32 var_458;
        wp::float32 var_459;
        wp::float32 var_460;
        const wp::int32 var_461 = 6;
        wp::int32* var_462;
        wp::int32 var_463;
        wp::int32 var_464;
        const wp::int32 var_465 = -1;
        bool var_466;
        wp::vec_t<3, wp::float32>* var_467;
        wp::vec_t<3, wp::float32> var_468;
        wp::vec_t<3, wp::float32> var_469;
        const wp::int32 var_470 = 1;
        wp::int32 var_471;
        wp::int32 var_472;
        const wp::int32 var_473 = 1;
        wp::int32 var_474;
        wp::int32 var_475;
        const wp::int32 var_476 = 2;
        bool var_477;
        const wp::int32 var_478 = 1;
        wp::int32 var_479;
        wp::float32 var_480;
        const wp::float32 var_481 = -1.0;
        const wp::float32 var_482 = 2.0;
        wp::float32 var_483;
        wp::float32 var_484;
        wp::float32 var_485;
        wp::float32 var_486;
        const wp::int32 var_487 = 2;
        bool var_488;
        const wp::int32 var_489 = 1;
        wp::int32 var_490;
        wp::float32 var_491;
        const wp::float32 var_492 = -1.0;
        const wp::float32 var_493 = 2.0;
        wp::float32 var_494;
        wp::float32 var_495;
        wp::float32 var_496;
        wp::float32 var_497;
        const wp::int32 var_498 = 2;
        bool var_499;
        const wp::int32 var_500 = 1;
        bool var_501;
        const wp::float32 var_502 = 1.0;
        const wp::float32 var_503 = 0.0;
        wp::float32 var_504;
        const wp::float32 var_505 = 0.5;
        wp::float32 var_506;
        const wp::int32 var_507 = 2;
        bool var_508;
        const wp::int32 var_509 = 1;
        bool var_510;
        const wp::float32 var_511 = 1.0;
        const wp::float32 var_512 = 0.0;
        wp::float32 var_513;
        const wp::float32 var_514 = 0.5;
        wp::float32 var_515;
        wp::float32 var_516;
        wp::float32 var_517;
        wp::vec_t<3, wp::float32> var_518;
        wp::vec_t<3, wp::float32> var_519;
        wp::vec_t<3, wp::float32> var_520;
        wp::vec_t<3, wp::float32> var_521;
        const wp::float32 var_522 = 0.0;
        wp::vec_t<3, wp::float32> var_523;
        wp::vec_t<3, wp::float32> var_524;
        wp::vec_t<3, wp::float32> var_525;
        wp::vec_t<3, wp::float32> var_526;
        wp::int32 var_527;
        wp::int32 var_528;
        wp::float32 var_529;
        wp::float32 var_530;
        wp::float32 var_531;
        wp::float32 var_532;
        wp::float32 var_533;
        wp::float32 var_534;
        const wp::int32 var_535 = 7;
        wp::int32* var_536;
        wp::int32 var_537;
        wp::int32 var_538;
        const wp::int32 var_539 = -1;
        bool var_540;
        wp::vec_t<3, wp::float32>* var_541;
        wp::vec_t<3, wp::float32> var_542;
        wp::vec_t<3, wp::float32> var_543;
        const wp::int32 var_544 = 1;
        wp::int32 var_545;
        wp::int32 var_546;
        const wp::int32 var_547 = 1;
        wp::int32 var_548;
        wp::int32 var_549;
        const wp::int32 var_550 = 2;
        bool var_551;
        const wp::int32 var_552 = 1;
        wp::int32 var_553;
        wp::float32 var_554;
        const wp::float32 var_555 = -1.0;
        const wp::float32 var_556 = 2.0;
        wp::float32 var_557;
        wp::float32 var_558;
        wp::float32 var_559;
        wp::float32 var_560;
        const wp::int32 var_561 = 2;
        bool var_562;
        const wp::int32 var_563 = 1;
        wp::int32 var_564;
        wp::float32 var_565;
        const wp::float32 var_566 = -1.0;
        const wp::float32 var_567 = 2.0;
        wp::float32 var_568;
        wp::float32 var_569;
        wp::float32 var_570;
        wp::float32 var_571;
        const wp::int32 var_572 = 2;
        bool var_573;
        const wp::int32 var_574 = 1;
        bool var_575;
        const wp::float32 var_576 = 1.0;
        const wp::float32 var_577 = 0.0;
        wp::float32 var_578;
        const wp::float32 var_579 = 0.5;
        wp::float32 var_580;
        const wp::int32 var_581 = 2;
        bool var_582;
        const wp::int32 var_583 = 1;
        bool var_584;
        const wp::float32 var_585 = 1.0;
        const wp::float32 var_586 = 0.0;
        wp::float32 var_587;
        const wp::float32 var_588 = 0.5;
        wp::float32 var_589;
        wp::float32 var_590;
        wp::float32 var_591;
        wp::vec_t<3, wp::float32> var_592;
        wp::vec_t<3, wp::float32> var_593;
        wp::vec_t<3, wp::float32> var_594;
        wp::vec_t<3, wp::float32> var_595;
        const wp::float32 var_596 = 0.0;
        wp::vec_t<3, wp::float32> var_597;
        wp::vec_t<3, wp::float32> var_598;
        wp::vec_t<3, wp::float32> var_599;
        wp::vec_t<3, wp::float32> var_600;
        wp::int32 var_601;
        wp::int32 var_602;
        wp::float32 var_603;
        wp::float32 var_604;
        wp::float32 var_605;
        wp::float32 var_606;
        wp::float32 var_607;
        wp::float32 var_608;
        const wp::int32 var_609 = 8;
        wp::int32* var_610;
        wp::int32 var_611;
        wp::int32 var_612;
        const wp::int32 var_613 = -1;
        bool var_614;
        wp::vec_t<3, wp::float32>* var_615;
        wp::vec_t<3, wp::float32> var_616;
        wp::vec_t<3, wp::float32> var_617;
        const wp::int32 var_618 = 1;
        wp::int32 var_619;
        wp::int32 var_620;
        const wp::int32 var_621 = 1;
        wp::int32 var_622;
        wp::int32 var_623;
        const wp::int32 var_624 = 2;
        bool var_625;
        const wp::int32 var_626 = 1;
        wp::int32 var_627;
        wp::float32 var_628;
        const wp::float32 var_629 = -1.0;
        const wp::float32 var_630 = 2.0;
        wp::float32 var_631;
        wp::float32 var_632;
        wp::float32 var_633;
        wp::float32 var_634;
        const wp::int32 var_635 = 2;
        bool var_636;
        const wp::int32 var_637 = 1;
        wp::int32 var_638;
        wp::float32 var_639;
        const wp::float32 var_640 = -1.0;
        const wp::float32 var_641 = 2.0;
        wp::float32 var_642;
        wp::float32 var_643;
        wp::float32 var_644;
        wp::float32 var_645;
        const wp::int32 var_646 = 2;
        bool var_647;
        const wp::int32 var_648 = 1;
        bool var_649;
        const wp::float32 var_650 = 1.0;
        const wp::float32 var_651 = 0.0;
        wp::float32 var_652;
        const wp::float32 var_653 = 0.5;
        wp::float32 var_654;
        const wp::int32 var_655 = 2;
        bool var_656;
        const wp::int32 var_657 = 1;
        bool var_658;
        const wp::float32 var_659 = 1.0;
        const wp::float32 var_660 = 0.0;
        wp::float32 var_661;
        const wp::float32 var_662 = 0.5;
        wp::float32 var_663;
        wp::float32 var_664;
        wp::float32 var_665;
        wp::vec_t<3, wp::float32> var_666;
        wp::vec_t<3, wp::float32> var_667;
        wp::vec_t<3, wp::float32> var_668;
        wp::vec_t<3, wp::float32> var_669;
        const wp::float32 var_670 = 0.0;
        wp::vec_t<3, wp::float32> var_671;
        wp::vec_t<3, wp::float32> var_672;
        wp::vec_t<3, wp::float32> var_673;
        wp::vec_t<3, wp::float32> var_674;
        wp::int32 var_675;
        wp::int32 var_676;
        wp::float32 var_677;
        wp::float32 var_678;
        wp::float32 var_679;
        wp::float32 var_680;
        wp::float32 var_681;
        wp::float32 var_682;
        wp::vec_t<3, wp::float32> var_683;
        wp::int32 var_684;
        wp::int32 var_685;
        wp::int32 var_686;
        wp::int32 var_687;
        wp::int32 var_688;
        wp::int32 var_689;
        const wp::float32 var_690 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_691;
        const wp::int32 var_692 = 0;
        bool var_693;
        const wp::int32 var_694 = 0;
        wp::float32 var_695;
        const wp::int32 var_696 = 0;
        wp::float32 var_697;
        const wp::int32 var_698 = 0;
        wp::float32 var_699;
        const wp::int32 var_700 = 1;
        wp::float32 var_701;
        const wp::int32 var_702 = 1;
        wp::float32 var_703;
        const wp::int32 var_704 = 1;
        wp::float32 var_705;
        const wp::int32 var_706 = 2;
        wp::float32 var_707;
        const wp::int32 var_708 = 2;
        wp::float32 var_709;
        const wp::int32 var_710 = 2;
        wp::float32 var_711;
        wp::mat_t<3, 3, wp::float32> var_712;
        const wp::int32 var_713 = 1;
        bool var_714;
        const wp::int32 var_715 = 0;
        wp::float32 var_716;
        const wp::int32 var_717 = 0;
        wp::float32 var_718;
        const wp::int32 var_719 = 0;
        wp::float32 var_720;
        const wp::int32 var_721 = 1;
        wp::float32 var_722;
        const wp::int32 var_723 = 1;
        wp::float32 var_724;
        const wp::int32 var_725 = 1;
        wp::float32 var_726;
        const wp::int32 var_727 = 2;
        wp::float32 var_728;
        const wp::int32 var_729 = 2;
        wp::float32 var_730;
        const wp::int32 var_731 = 2;
        wp::float32 var_732;
        wp::mat_t<3, 3, wp::float32> var_733;
        const wp::int32 var_734 = 0;
        wp::float32 var_735;
        const wp::int32 var_736 = 0;
        wp::float32 var_737;
        const wp::int32 var_738 = 0;
        wp::float32 var_739;
        const wp::int32 var_740 = 1;
        wp::float32 var_741;
        const wp::int32 var_742 = 1;
        wp::float32 var_743;
        const wp::int32 var_744 = 1;
        wp::float32 var_745;
        const wp::int32 var_746 = 2;
        wp::float32 var_747;
        const wp::int32 var_748 = 2;
        wp::float32 var_749;
        const wp::int32 var_750 = 2;
        wp::float32 var_751;
        wp::mat_t<3, 3, wp::float32> var_752;
        wp::mat_t<3, 3, wp::float32> var_753;
        wp::mat_t<3, 3, wp::float32> var_754;
        wp::quat_t<wp::float32> var_755;
        //---------
        // forward
        // def _flex_face_kinematics(                                                             <L 509>
        // worldid, face_id = wp.tid()                                                            <L 521>
        builtin_tid2d(var_0, var_1);
        // mapping = flex_face_map[face_id]                                                       <L 523>
        var_2 = wp::address(var_flex_face_map, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // f = mapping[0]                                                                         <L 524>
        var_6 = wp::extract(var_3, var_5);
        // face_elem_idx = mapping[1]                                                             <L 525>
        var_8 = wp::extract(var_3, var_7);
        // order = flex_interp[f]                                                                 <L 527>
        var_9 = wp::address(var_flex_interp, var_6);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // order_abs = -order                                                                     <L 528>
        var_12 = wp::neg(var_10);
        // cellnum = flex_cellnum[f]                                                              <L 529>
        var_13 = wp::address(var_flex_cellnum, var_6);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // cx = cellnum[0]                                                                        <L 530>
        var_17 = wp::extract(var_14, var_16);
        // cy = cellnum[1]                                                                        <L 531>
        var_19 = wp::extract(var_14, var_18);
        // cz = cellnum[2]                                                                        <L 532>
        var_21 = wp::extract(var_14, var_20);
        // t1 = wp.vec3(0.0)                                                                      <L 533>
        var_23 = wp::vec_t<3, wp::float32>(var_22);
        // t2 = wp.vec3(0.0)                                                                      <L 534>
        var_25 = wp::vec_t<3, wp::float32>(var_24);
        // for local_idx in range(9):                                                             <L 535>
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_27 = wp::address(var_flex_face, var_1, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // if gidx != -1:                                                                         <L 537>
        var_31 = (var_28 != var_30);
        if (var_31) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_32 = wp::address(var_flexnode_xpos_in, var_0, var_28);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_26, var_33);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_36 = wp::add(var_12, var_35);
            var_37 = wp::floordiv(var_26, var_36);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_39 = wp::add(var_12, var_38);
            var_40 = wp::mod(var_26, var_39);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_42 = (var_12 == var_41);
            if (var_42) {
                var_44 = wp::sub(var_37, var_43);
                var_45 = wp::float(var_44);
            }
            if (!var_42) {
                var_48 = wp::float(var_37);
                var_49 = wp::mul(var_47, var_48);
                var_50 = wp::add(var_46, var_49);
            }
            var_51 = wp::where(var_42, var_45, var_50);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_53 = (var_12 == var_52);
            if (var_53) {
                var_55 = wp::sub(var_40, var_54);
                var_56 = wp::float(var_55);
            }
            if (!var_53) {
                var_59 = wp::float(var_40);
                var_60 = wp::mul(var_58, var_59);
                var_61 = wp::add(var_57, var_60);
            }
            var_62 = wp::where(var_53, var_56, var_61);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_64 = (var_12 == var_63);
            if (var_64) {
                var_66 = (var_37 == var_65);
                var_69 = wp::where(var_66, var_67, var_68);
            }
            if (!var_64) {
            }
            var_71 = wp::where(var_64, var_69, var_70);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_73 = (var_12 == var_72);
            if (var_73) {
                var_75 = (var_40 == var_74);
                var_78 = wp::where(var_75, var_76, var_77);
            }
            if (!var_73) {
            }
            var_80 = wp::where(var_73, var_78, var_79);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_81 = wp::mul(var_51, var_80);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_82 = wp::mul(var_71, var_62);
            // t1 += node_pos * grad0                                                             <L 553>
            var_83 = wp::mul(var_33, var_81);
            var_84 = wp::add(var_23, var_83);
            // t2 += node_pos * grad1                                                             <L 554>
            var_85 = wp::mul(var_33, var_82);
            var_86 = wp::add(var_25, var_85);
        }
        if (!var_31) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_88 = wp::vec_t<3, wp::float32>(var_87);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_26, var_88);
        }
        var_89 = wp::where(var_31, var_84, var_23);
        var_90 = wp::where(var_31, var_86, var_25);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_92 = wp::address(var_flex_face, var_1, var_91);
        var_94 = wp::load(var_92);
        var_93 = wp::copy(var_94);
        // if gidx != -1:                                                                         <L 537>
        var_96 = (var_93 != var_95);
        if (var_96) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_97 = wp::address(var_flexnode_xpos_in, var_0, var_93);
            var_99 = wp::load(var_97);
            var_98 = wp::copy(var_99);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_91, var_98);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_101 = wp::add(var_12, var_100);
            var_102 = wp::floordiv(var_91, var_101);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_104 = wp::add(var_12, var_103);
            var_105 = wp::mod(var_91, var_104);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_107 = (var_12 == var_106);
            if (var_107) {
                var_109 = wp::sub(var_102, var_108);
                var_110 = wp::float(var_109);
            }
            if (!var_107) {
                var_113 = wp::float(var_102);
                var_114 = wp::mul(var_112, var_113);
                var_115 = wp::add(var_111, var_114);
            }
            var_116 = wp::where(var_107, var_110, var_115);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_118 = (var_12 == var_117);
            if (var_118) {
                var_120 = wp::sub(var_105, var_119);
                var_121 = wp::float(var_120);
            }
            if (!var_118) {
                var_124 = wp::float(var_105);
                var_125 = wp::mul(var_123, var_124);
                var_126 = wp::add(var_122, var_125);
            }
            var_127 = wp::where(var_118, var_121, var_126);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_129 = (var_12 == var_128);
            if (var_129) {
                var_131 = (var_102 == var_130);
                var_134 = wp::where(var_131, var_132, var_133);
            }
            if (!var_129) {
            }
            var_136 = wp::where(var_129, var_134, var_135);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_138 = (var_12 == var_137);
            if (var_138) {
                var_140 = (var_105 == var_139);
                var_143 = wp::where(var_140, var_141, var_142);
            }
            if (!var_138) {
            }
            var_145 = wp::where(var_138, var_143, var_144);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_146 = wp::mul(var_116, var_145);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_147 = wp::mul(var_136, var_127);
            // t1 += node_pos * grad0                                                             <L 553>
            var_148 = wp::mul(var_98, var_146);
            var_149 = wp::add(var_89, var_148);
            // t2 += node_pos * grad1                                                             <L 554>
            var_150 = wp::mul(var_98, var_147);
            var_151 = wp::add(var_90, var_150);
        }
        if (!var_96) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_153 = wp::vec_t<3, wp::float32>(var_152);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_91, var_153);
        }
        var_154 = wp::where(var_96, var_149, var_89);
        var_155 = wp::where(var_96, var_151, var_90);
        var_156 = wp::where(var_96, var_98, var_33);
        var_157 = wp::where(var_96, var_102, var_37);
        var_158 = wp::where(var_96, var_105, var_40);
        var_159 = wp::where(var_96, var_116, var_51);
        var_160 = wp::where(var_96, var_127, var_62);
        var_161 = wp::where(var_96, var_136, var_71);
        var_162 = wp::where(var_96, var_145, var_80);
        var_163 = wp::where(var_96, var_146, var_81);
        var_164 = wp::where(var_96, var_147, var_82);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_166 = wp::address(var_flex_face, var_1, var_165);
        var_168 = wp::load(var_166);
        var_167 = wp::copy(var_168);
        // if gidx != -1:                                                                         <L 537>
        var_170 = (var_167 != var_169);
        if (var_170) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_171 = wp::address(var_flexnode_xpos_in, var_0, var_167);
            var_173 = wp::load(var_171);
            var_172 = wp::copy(var_173);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_165, var_172);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_175 = wp::add(var_12, var_174);
            var_176 = wp::floordiv(var_165, var_175);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_178 = wp::add(var_12, var_177);
            var_179 = wp::mod(var_165, var_178);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_181 = (var_12 == var_180);
            if (var_181) {
                var_183 = wp::sub(var_176, var_182);
                var_184 = wp::float(var_183);
            }
            if (!var_181) {
                var_187 = wp::float(var_176);
                var_188 = wp::mul(var_186, var_187);
                var_189 = wp::add(var_185, var_188);
            }
            var_190 = wp::where(var_181, var_184, var_189);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_192 = (var_12 == var_191);
            if (var_192) {
                var_194 = wp::sub(var_179, var_193);
                var_195 = wp::float(var_194);
            }
            if (!var_192) {
                var_198 = wp::float(var_179);
                var_199 = wp::mul(var_197, var_198);
                var_200 = wp::add(var_196, var_199);
            }
            var_201 = wp::where(var_192, var_195, var_200);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_203 = (var_12 == var_202);
            if (var_203) {
                var_205 = (var_176 == var_204);
                var_208 = wp::where(var_205, var_206, var_207);
            }
            if (!var_203) {
            }
            var_210 = wp::where(var_203, var_208, var_209);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_212 = (var_12 == var_211);
            if (var_212) {
                var_214 = (var_179 == var_213);
                var_217 = wp::where(var_214, var_215, var_216);
            }
            if (!var_212) {
            }
            var_219 = wp::where(var_212, var_217, var_218);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_220 = wp::mul(var_190, var_219);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_221 = wp::mul(var_210, var_201);
            // t1 += node_pos * grad0                                                             <L 553>
            var_222 = wp::mul(var_172, var_220);
            var_223 = wp::add(var_154, var_222);
            // t2 += node_pos * grad1                                                             <L 554>
            var_224 = wp::mul(var_172, var_221);
            var_225 = wp::add(var_155, var_224);
        }
        if (!var_170) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_227 = wp::vec_t<3, wp::float32>(var_226);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_165, var_227);
        }
        var_228 = wp::where(var_170, var_223, var_154);
        var_229 = wp::where(var_170, var_225, var_155);
        var_230 = wp::where(var_170, var_172, var_156);
        var_231 = wp::where(var_170, var_176, var_157);
        var_232 = wp::where(var_170, var_179, var_158);
        var_233 = wp::where(var_170, var_190, var_159);
        var_234 = wp::where(var_170, var_201, var_160);
        var_235 = wp::where(var_170, var_210, var_161);
        var_236 = wp::where(var_170, var_219, var_162);
        var_237 = wp::where(var_170, var_220, var_163);
        var_238 = wp::where(var_170, var_221, var_164);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_240 = wp::address(var_flex_face, var_1, var_239);
        var_242 = wp::load(var_240);
        var_241 = wp::copy(var_242);
        // if gidx != -1:                                                                         <L 537>
        var_244 = (var_241 != var_243);
        if (var_244) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_245 = wp::address(var_flexnode_xpos_in, var_0, var_241);
            var_247 = wp::load(var_245);
            var_246 = wp::copy(var_247);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_239, var_246);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_249 = wp::add(var_12, var_248);
            var_250 = wp::floordiv(var_239, var_249);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_252 = wp::add(var_12, var_251);
            var_253 = wp::mod(var_239, var_252);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_255 = (var_12 == var_254);
            if (var_255) {
                var_257 = wp::sub(var_250, var_256);
                var_258 = wp::float(var_257);
            }
            if (!var_255) {
                var_261 = wp::float(var_250);
                var_262 = wp::mul(var_260, var_261);
                var_263 = wp::add(var_259, var_262);
            }
            var_264 = wp::where(var_255, var_258, var_263);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_266 = (var_12 == var_265);
            if (var_266) {
                var_268 = wp::sub(var_253, var_267);
                var_269 = wp::float(var_268);
            }
            if (!var_266) {
                var_272 = wp::float(var_253);
                var_273 = wp::mul(var_271, var_272);
                var_274 = wp::add(var_270, var_273);
            }
            var_275 = wp::where(var_266, var_269, var_274);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_277 = (var_12 == var_276);
            if (var_277) {
                var_279 = (var_250 == var_278);
                var_282 = wp::where(var_279, var_280, var_281);
            }
            if (!var_277) {
            }
            var_284 = wp::where(var_277, var_282, var_283);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_286 = (var_12 == var_285);
            if (var_286) {
                var_288 = (var_253 == var_287);
                var_291 = wp::where(var_288, var_289, var_290);
            }
            if (!var_286) {
            }
            var_293 = wp::where(var_286, var_291, var_292);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_294 = wp::mul(var_264, var_293);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_295 = wp::mul(var_284, var_275);
            // t1 += node_pos * grad0                                                             <L 553>
            var_296 = wp::mul(var_246, var_294);
            var_297 = wp::add(var_228, var_296);
            // t2 += node_pos * grad1                                                             <L 554>
            var_298 = wp::mul(var_246, var_295);
            var_299 = wp::add(var_229, var_298);
        }
        if (!var_244) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_301 = wp::vec_t<3, wp::float32>(var_300);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_239, var_301);
        }
        var_302 = wp::where(var_244, var_297, var_228);
        var_303 = wp::where(var_244, var_299, var_229);
        var_304 = wp::where(var_244, var_246, var_230);
        var_305 = wp::where(var_244, var_250, var_231);
        var_306 = wp::where(var_244, var_253, var_232);
        var_307 = wp::where(var_244, var_264, var_233);
        var_308 = wp::where(var_244, var_275, var_234);
        var_309 = wp::where(var_244, var_284, var_235);
        var_310 = wp::where(var_244, var_293, var_236);
        var_311 = wp::where(var_244, var_294, var_237);
        var_312 = wp::where(var_244, var_295, var_238);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_314 = wp::address(var_flex_face, var_1, var_313);
        var_316 = wp::load(var_314);
        var_315 = wp::copy(var_316);
        // if gidx != -1:                                                                         <L 537>
        var_318 = (var_315 != var_317);
        if (var_318) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_319 = wp::address(var_flexnode_xpos_in, var_0, var_315);
            var_321 = wp::load(var_319);
            var_320 = wp::copy(var_321);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_313, var_320);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_323 = wp::add(var_12, var_322);
            var_324 = wp::floordiv(var_313, var_323);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_326 = wp::add(var_12, var_325);
            var_327 = wp::mod(var_313, var_326);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_329 = (var_12 == var_328);
            if (var_329) {
                var_331 = wp::sub(var_324, var_330);
                var_332 = wp::float(var_331);
            }
            if (!var_329) {
                var_335 = wp::float(var_324);
                var_336 = wp::mul(var_334, var_335);
                var_337 = wp::add(var_333, var_336);
            }
            var_338 = wp::where(var_329, var_332, var_337);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_340 = (var_12 == var_339);
            if (var_340) {
                var_342 = wp::sub(var_327, var_341);
                var_343 = wp::float(var_342);
            }
            if (!var_340) {
                var_346 = wp::float(var_327);
                var_347 = wp::mul(var_345, var_346);
                var_348 = wp::add(var_344, var_347);
            }
            var_349 = wp::where(var_340, var_343, var_348);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_351 = (var_12 == var_350);
            if (var_351) {
                var_353 = (var_324 == var_352);
                var_356 = wp::where(var_353, var_354, var_355);
            }
            if (!var_351) {
            }
            var_358 = wp::where(var_351, var_356, var_357);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_360 = (var_12 == var_359);
            if (var_360) {
                var_362 = (var_327 == var_361);
                var_365 = wp::where(var_362, var_363, var_364);
            }
            if (!var_360) {
            }
            var_367 = wp::where(var_360, var_365, var_366);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_368 = wp::mul(var_338, var_367);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_369 = wp::mul(var_358, var_349);
            // t1 += node_pos * grad0                                                             <L 553>
            var_370 = wp::mul(var_320, var_368);
            var_371 = wp::add(var_302, var_370);
            // t2 += node_pos * grad1                                                             <L 554>
            var_372 = wp::mul(var_320, var_369);
            var_373 = wp::add(var_303, var_372);
        }
        if (!var_318) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_375 = wp::vec_t<3, wp::float32>(var_374);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_313, var_375);
        }
        var_376 = wp::where(var_318, var_371, var_302);
        var_377 = wp::where(var_318, var_373, var_303);
        var_378 = wp::where(var_318, var_320, var_304);
        var_379 = wp::where(var_318, var_324, var_305);
        var_380 = wp::where(var_318, var_327, var_306);
        var_381 = wp::where(var_318, var_338, var_307);
        var_382 = wp::where(var_318, var_349, var_308);
        var_383 = wp::where(var_318, var_358, var_309);
        var_384 = wp::where(var_318, var_367, var_310);
        var_385 = wp::where(var_318, var_368, var_311);
        var_386 = wp::where(var_318, var_369, var_312);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_388 = wp::address(var_flex_face, var_1, var_387);
        var_390 = wp::load(var_388);
        var_389 = wp::copy(var_390);
        // if gidx != -1:                                                                         <L 537>
        var_392 = (var_389 != var_391);
        if (var_392) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_393 = wp::address(var_flexnode_xpos_in, var_0, var_389);
            var_395 = wp::load(var_393);
            var_394 = wp::copy(var_395);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_387, var_394);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_397 = wp::add(var_12, var_396);
            var_398 = wp::floordiv(var_387, var_397);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_400 = wp::add(var_12, var_399);
            var_401 = wp::mod(var_387, var_400);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_403 = (var_12 == var_402);
            if (var_403) {
                var_405 = wp::sub(var_398, var_404);
                var_406 = wp::float(var_405);
            }
            if (!var_403) {
                var_409 = wp::float(var_398);
                var_410 = wp::mul(var_408, var_409);
                var_411 = wp::add(var_407, var_410);
            }
            var_412 = wp::where(var_403, var_406, var_411);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_414 = (var_12 == var_413);
            if (var_414) {
                var_416 = wp::sub(var_401, var_415);
                var_417 = wp::float(var_416);
            }
            if (!var_414) {
                var_420 = wp::float(var_401);
                var_421 = wp::mul(var_419, var_420);
                var_422 = wp::add(var_418, var_421);
            }
            var_423 = wp::where(var_414, var_417, var_422);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_425 = (var_12 == var_424);
            if (var_425) {
                var_427 = (var_398 == var_426);
                var_430 = wp::where(var_427, var_428, var_429);
            }
            if (!var_425) {
            }
            var_432 = wp::where(var_425, var_430, var_431);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_434 = (var_12 == var_433);
            if (var_434) {
                var_436 = (var_401 == var_435);
                var_439 = wp::where(var_436, var_437, var_438);
            }
            if (!var_434) {
            }
            var_441 = wp::where(var_434, var_439, var_440);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_442 = wp::mul(var_412, var_441);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_443 = wp::mul(var_432, var_423);
            // t1 += node_pos * grad0                                                             <L 553>
            var_444 = wp::mul(var_394, var_442);
            var_445 = wp::add(var_376, var_444);
            // t2 += node_pos * grad1                                                             <L 554>
            var_446 = wp::mul(var_394, var_443);
            var_447 = wp::add(var_377, var_446);
        }
        if (!var_392) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_449 = wp::vec_t<3, wp::float32>(var_448);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_387, var_449);
        }
        var_450 = wp::where(var_392, var_445, var_376);
        var_451 = wp::where(var_392, var_447, var_377);
        var_452 = wp::where(var_392, var_394, var_378);
        var_453 = wp::where(var_392, var_398, var_379);
        var_454 = wp::where(var_392, var_401, var_380);
        var_455 = wp::where(var_392, var_412, var_381);
        var_456 = wp::where(var_392, var_423, var_382);
        var_457 = wp::where(var_392, var_432, var_383);
        var_458 = wp::where(var_392, var_441, var_384);
        var_459 = wp::where(var_392, var_442, var_385);
        var_460 = wp::where(var_392, var_443, var_386);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_462 = wp::address(var_flex_face, var_1, var_461);
        var_464 = wp::load(var_462);
        var_463 = wp::copy(var_464);
        // if gidx != -1:                                                                         <L 537>
        var_466 = (var_463 != var_465);
        if (var_466) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_467 = wp::address(var_flexnode_xpos_in, var_0, var_463);
            var_469 = wp::load(var_467);
            var_468 = wp::copy(var_469);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_461, var_468);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_471 = wp::add(var_12, var_470);
            var_472 = wp::floordiv(var_461, var_471);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_474 = wp::add(var_12, var_473);
            var_475 = wp::mod(var_461, var_474);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_477 = (var_12 == var_476);
            if (var_477) {
                var_479 = wp::sub(var_472, var_478);
                var_480 = wp::float(var_479);
            }
            if (!var_477) {
                var_483 = wp::float(var_472);
                var_484 = wp::mul(var_482, var_483);
                var_485 = wp::add(var_481, var_484);
            }
            var_486 = wp::where(var_477, var_480, var_485);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_488 = (var_12 == var_487);
            if (var_488) {
                var_490 = wp::sub(var_475, var_489);
                var_491 = wp::float(var_490);
            }
            if (!var_488) {
                var_494 = wp::float(var_475);
                var_495 = wp::mul(var_493, var_494);
                var_496 = wp::add(var_492, var_495);
            }
            var_497 = wp::where(var_488, var_491, var_496);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_499 = (var_12 == var_498);
            if (var_499) {
                var_501 = (var_472 == var_500);
                var_504 = wp::where(var_501, var_502, var_503);
            }
            if (!var_499) {
            }
            var_506 = wp::where(var_499, var_504, var_505);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_508 = (var_12 == var_507);
            if (var_508) {
                var_510 = (var_475 == var_509);
                var_513 = wp::where(var_510, var_511, var_512);
            }
            if (!var_508) {
            }
            var_515 = wp::where(var_508, var_513, var_514);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_516 = wp::mul(var_486, var_515);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_517 = wp::mul(var_506, var_497);
            // t1 += node_pos * grad0                                                             <L 553>
            var_518 = wp::mul(var_468, var_516);
            var_519 = wp::add(var_450, var_518);
            // t2 += node_pos * grad1                                                             <L 554>
            var_520 = wp::mul(var_468, var_517);
            var_521 = wp::add(var_451, var_520);
        }
        if (!var_466) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_523 = wp::vec_t<3, wp::float32>(var_522);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_461, var_523);
        }
        var_524 = wp::where(var_466, var_519, var_450);
        var_525 = wp::where(var_466, var_521, var_451);
        var_526 = wp::where(var_466, var_468, var_452);
        var_527 = wp::where(var_466, var_472, var_453);
        var_528 = wp::where(var_466, var_475, var_454);
        var_529 = wp::where(var_466, var_486, var_455);
        var_530 = wp::where(var_466, var_497, var_456);
        var_531 = wp::where(var_466, var_506, var_457);
        var_532 = wp::where(var_466, var_515, var_458);
        var_533 = wp::where(var_466, var_516, var_459);
        var_534 = wp::where(var_466, var_517, var_460);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_536 = wp::address(var_flex_face, var_1, var_535);
        var_538 = wp::load(var_536);
        var_537 = wp::copy(var_538);
        // if gidx != -1:                                                                         <L 537>
        var_540 = (var_537 != var_539);
        if (var_540) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_541 = wp::address(var_flexnode_xpos_in, var_0, var_537);
            var_543 = wp::load(var_541);
            var_542 = wp::copy(var_543);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_535, var_542);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_545 = wp::add(var_12, var_544);
            var_546 = wp::floordiv(var_535, var_545);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_548 = wp::add(var_12, var_547);
            var_549 = wp::mod(var_535, var_548);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_551 = (var_12 == var_550);
            if (var_551) {
                var_553 = wp::sub(var_546, var_552);
                var_554 = wp::float(var_553);
            }
            if (!var_551) {
                var_557 = wp::float(var_546);
                var_558 = wp::mul(var_556, var_557);
                var_559 = wp::add(var_555, var_558);
            }
            var_560 = wp::where(var_551, var_554, var_559);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_562 = (var_12 == var_561);
            if (var_562) {
                var_564 = wp::sub(var_549, var_563);
                var_565 = wp::float(var_564);
            }
            if (!var_562) {
                var_568 = wp::float(var_549);
                var_569 = wp::mul(var_567, var_568);
                var_570 = wp::add(var_566, var_569);
            }
            var_571 = wp::where(var_562, var_565, var_570);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_573 = (var_12 == var_572);
            if (var_573) {
                var_575 = (var_546 == var_574);
                var_578 = wp::where(var_575, var_576, var_577);
            }
            if (!var_573) {
            }
            var_580 = wp::where(var_573, var_578, var_579);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_582 = (var_12 == var_581);
            if (var_582) {
                var_584 = (var_549 == var_583);
                var_587 = wp::where(var_584, var_585, var_586);
            }
            if (!var_582) {
            }
            var_589 = wp::where(var_582, var_587, var_588);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_590 = wp::mul(var_560, var_589);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_591 = wp::mul(var_580, var_571);
            // t1 += node_pos * grad0                                                             <L 553>
            var_592 = wp::mul(var_542, var_590);
            var_593 = wp::add(var_524, var_592);
            // t2 += node_pos * grad1                                                             <L 554>
            var_594 = wp::mul(var_542, var_591);
            var_595 = wp::add(var_525, var_594);
        }
        if (!var_540) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_597 = wp::vec_t<3, wp::float32>(var_596);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_535, var_597);
        }
        var_598 = wp::where(var_540, var_593, var_524);
        var_599 = wp::where(var_540, var_595, var_525);
        var_600 = wp::where(var_540, var_542, var_526);
        var_601 = wp::where(var_540, var_546, var_527);
        var_602 = wp::where(var_540, var_549, var_528);
        var_603 = wp::where(var_540, var_560, var_529);
        var_604 = wp::where(var_540, var_571, var_530);
        var_605 = wp::where(var_540, var_580, var_531);
        var_606 = wp::where(var_540, var_589, var_532);
        var_607 = wp::where(var_540, var_590, var_533);
        var_608 = wp::where(var_540, var_591, var_534);
        // gidx = flex_face[face_id, local_idx]                                                   <L 536>
        var_610 = wp::address(var_flex_face, var_1, var_609);
        var_612 = wp::load(var_610);
        var_611 = wp::copy(var_612);
        // if gidx != -1:                                                                         <L 537>
        var_614 = (var_611 != var_613);
        if (var_614) {
            // node_pos = flexnode_xpos_in[worldid, gidx]                                         <L 538>
            var_615 = wp::address(var_flexnode_xpos_in, var_0, var_611);
            var_617 = wp::load(var_615);
            var_616 = wp::copy(var_617);
            // face_xpos_out[worldid, face_id, local_idx] = node_pos                              <L 540>
            wp::array_store(var_face_xpos_out, var_0, var_1, var_609, var_616);
            // l0 = local_idx // (order_abs + 1)                                                  <L 542>
            var_619 = wp::add(var_12, var_618);
            var_620 = wp::floordiv(var_609, var_619);
            // l1 = local_idx % (order_abs + 1)                                                   <L 543>
            var_622 = wp::add(var_12, var_621);
            var_623 = wp::mod(var_609, var_622);
            // dphi0 = float(l0 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l0))              <L 545>
            var_625 = (var_12 == var_624);
            if (var_625) {
                var_627 = wp::sub(var_620, var_626);
                var_628 = wp::float(var_627);
            }
            if (!var_625) {
                var_631 = wp::float(var_620);
                var_632 = wp::mul(var_630, var_631);
                var_633 = wp::add(var_629, var_632);
            }
            var_634 = wp::where(var_625, var_628, var_633);
            // dphi1 = float(l1 - 1) if order_abs == 2 else (-1.0 + 2.0 * float(l1))              <L 546>
            var_636 = (var_12 == var_635);
            if (var_636) {
                var_638 = wp::sub(var_623, var_637);
                var_639 = wp::float(var_638);
            }
            if (!var_636) {
                var_642 = wp::float(var_623);
                var_643 = wp::mul(var_641, var_642);
                var_644 = wp::add(var_640, var_643);
            }
            var_645 = wp::where(var_636, var_639, var_644);
            // phi0 = wp.where(l0 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 547>
            var_647 = (var_12 == var_646);
            if (var_647) {
                var_649 = (var_620 == var_648);
                var_652 = wp::where(var_649, var_650, var_651);
            }
            if (!var_647) {
            }
            var_654 = wp::where(var_647, var_652, var_653);
            // phi1 = wp.where(l1 == 1, 1.0, 0.0) if order_abs == 2 else 0.5                      <L 548>
            var_656 = (var_12 == var_655);
            if (var_656) {
                var_658 = (var_623 == var_657);
                var_661 = wp::where(var_658, var_659, var_660);
            }
            if (!var_656) {
            }
            var_663 = wp::where(var_656, var_661, var_662);
            // grad0 = dphi0 * phi1                                                               <L 550>
            var_664 = wp::mul(var_634, var_663);
            // grad1 = phi0 * dphi1                                                               <L 551>
            var_665 = wp::mul(var_654, var_645);
            // t1 += node_pos * grad0                                                             <L 553>
            var_666 = wp::mul(var_616, var_664);
            var_667 = wp::add(var_598, var_666);
            // t2 += node_pos * grad1                                                             <L 554>
            var_668 = wp::mul(var_616, var_665);
            var_669 = wp::add(var_599, var_668);
        }
        if (!var_614) {
            // face_xpos_out[worldid, face_id, local_idx] = wp.vec3(0.0)                          <L 556>
            var_671 = wp::vec_t<3, wp::float32>(var_670);
            wp::array_store(var_face_xpos_out, var_0, var_1, var_609, var_671);
        }
        var_672 = wp::where(var_614, var_667, var_598);
        var_673 = wp::where(var_614, var_669, var_599);
        var_674 = wp::where(var_614, var_616, var_600);
        var_675 = wp::where(var_614, var_620, var_601);
        var_676 = wp::where(var_614, var_623, var_602);
        var_677 = wp::where(var_614, var_634, var_603);
        var_678 = wp::where(var_614, var_645, var_604);
        var_679 = wp::where(var_614, var_654, var_605);
        var_680 = wp::where(var_614, var_663, var_606);
        var_681 = wp::where(var_614, var_664, var_607);
        var_682 = wp::where(var_614, var_665, var_608);
        // normal = wp.cross(t1, t2)                                                              <L 558>
        var_683 = wp::cross(var_672, var_673);
        // normal_axis, _, _, _, _, _ = support.get_face_metadata(cx, cy, cz, face_elem_idx, order_abs)       <L 560>
        get_face_metadata_0(var_17, var_19, var_21, var_8, var_12, var_684, var_685, var_686, var_687, var_688, var_689);
        // F = wp.mat33(0.0)                                                                      <L 562>
        var_691 = wp::mat_t<3, 3, wp::float32>(var_690);
        // if normal_axis == 0:                                                                   <L 563>
        var_693 = (var_684 == var_692);
        if (var_693) {
            // F = wp.mat33(                                                                      <L 564>
            // normal[0],                                                                         <L 565>
            var_695 = wp::extract(var_683, var_694);
            // t1[0],                                                                             <L 566>
            var_697 = wp::extract(var_672, var_696);
            // t2[0],                                                                             <L 567>
            var_699 = wp::extract(var_673, var_698);
            // normal[1],                                                                         <L 568>
            var_701 = wp::extract(var_683, var_700);
            // t1[1],                                                                             <L 569>
            var_703 = wp::extract(var_672, var_702);
            // t2[1],                                                                             <L 570>
            var_705 = wp::extract(var_673, var_704);
            // normal[2],                                                                         <L 571>
            var_707 = wp::extract(var_683, var_706);
            // t1[2],                                                                             <L 572>
            var_709 = wp::extract(var_672, var_708);
            // t2[2],                                                                             <L 573>
            var_711 = wp::extract(var_673, var_710);
            var_712 = wp::mat_t<3, 3, wp::float32>(var_695, var_697, var_699, var_701, var_703, var_705, var_707, var_709, var_711);
        }
        if (!var_693) {
            // elif normal_axis == 1:                                                             <L 575>
            var_714 = (var_684 == var_713);
            if (var_714) {
                // F = wp.mat33(                                                                  <L 576>
                // t2[0],                                                                         <L 577>
                var_716 = wp::extract(var_673, var_715);
                // normal[0],                                                                     <L 578>
                var_718 = wp::extract(var_683, var_717);
                // t1[0],                                                                         <L 579>
                var_720 = wp::extract(var_672, var_719);
                // t2[1],                                                                         <L 580>
                var_722 = wp::extract(var_673, var_721);
                // normal[1],                                                                     <L 581>
                var_724 = wp::extract(var_683, var_723);
                // t1[1],                                                                         <L 582>
                var_726 = wp::extract(var_672, var_725);
                // t2[2],                                                                         <L 583>
                var_728 = wp::extract(var_673, var_727);
                // normal[2],                                                                     <L 584>
                var_730 = wp::extract(var_683, var_729);
                // t1[2],                                                                         <L 585>
                var_732 = wp::extract(var_672, var_731);
                var_733 = wp::mat_t<3, 3, wp::float32>(var_716, var_718, var_720, var_722, var_724, var_726, var_728, var_730, var_732);
            }
            if (!var_714) {
                // F = wp.mat33(                                                                  <L 588>
                // t1[0],                                                                         <L 589>
                var_735 = wp::extract(var_672, var_734);
                // t2[0],                                                                         <L 590>
                var_737 = wp::extract(var_673, var_736);
                // normal[0],                                                                     <L 591>
                var_739 = wp::extract(var_683, var_738);
                // t1[1],                                                                         <L 592>
                var_741 = wp::extract(var_672, var_740);
                // t2[1],                                                                         <L 593>
                var_743 = wp::extract(var_673, var_742);
                // normal[1],                                                                     <L 594>
                var_745 = wp::extract(var_683, var_744);
                // t1[2],                                                                         <L 595>
                var_747 = wp::extract(var_672, var_746);
                // t2[2],                                                                         <L 596>
                var_749 = wp::extract(var_673, var_748);
                // normal[2],                                                                     <L 597>
                var_751 = wp::extract(var_683, var_750);
                var_752 = wp::mat_t<3, 3, wp::float32>(var_735, var_737, var_739, var_741, var_743, var_745, var_747, var_749, var_751);
            }
            var_753 = wp::where(var_714, var_733, var_752);
        }
        var_754 = wp::where(var_693, var_712, var_753);
        // face_quat_out[worldid, face_id] = support.mat33_to_quat_polar(F)                       <L 600>
        var_755 = mat33_to_quat_polar_0(var_754);
        wp::array_store(var_face_quat_out, var_0, var_1, var_755);
    }
}



extern "C" __global__ void _cfrc_ext_contact_a1976729_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::vec_t<2, wp::int32>* var_5;
        wp::vec_t<2, wp::int32> var_6;
        wp::vec_t<2, wp::int32> var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        bool var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const bool var_26 = true;
        wp::vec_t<6, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::int32* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::int32 var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::array_t<wp::vec_t<6, wp::float32>> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<6, wp::float32> var_38;
        wp::vec_t<6, wp::float32> var_39;
        wp::int32* var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::array_t<wp::vec_t<6, wp::float32>> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        //---------
        // forward
        // def _cfrc_ext_contact(                                                                 <L 1680>
        // contactid = wp.tid()                                                                   <L 1700>
        var_0 = builtin_tid1d();
        // if contactid >= nacon_in[0]:                                                           <L 1702>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 1703>
            continue;
        }
        // geom = contact_geom_in[contactid]                                                      <L 1705>
        var_5 = wp::address(var_contact_geom_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // id1 = geom_bodyid[geom[0]]                                                             <L 1706>
        var_9 = wp::extract(var_6, var_8);
        var_10 = wp::address(var_geom_bodyid, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // id2 = geom_bodyid[geom[1]]                                                             <L 1707>
        var_14 = wp::extract(var_6, var_13);
        var_15 = wp::address(var_geom_bodyid, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // if id1 == 0 and id2 == 0:                                                              <L 1709>
        var_20 = (var_11 == var_19);
        var_18 = var_20;
        if (var_18) {
            var_22 = (var_16 == var_21);
            var_18 = var_18 && var_22;
        }
        if (var_18) {
            // return                                                                             <L 1710>
            continue;
        }
        // worldid = contact_worldid_in[contactid]                                                <L 1712>
        var_23 = wp::address(var_contact_worldid_in, var_0);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // force = support.contact_force_fn(                                                      <L 1715>
        // opt_cone,                                                                              <L 1716>
        // contact_frame_in,                                                                      <L 1717>
        // contact_friction_in,                                                                   <L 1718>
        // contact_dim_in,                                                                        <L 1719>
        // contact_efc_address_in,                                                                <L 1720>
        // efc_force_in,                                                                          <L 1721>
        // njmax_in,                                                                              <L 1722>
        // nacon_in,                                                                              <L 1723>
        // worldid,                                                                               <L 1724>
        // contactid,                                                                             <L 1725>
        // to_world_frame=True,                                                                   <L 1726>
        var_27 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_24, var_0, var_26);
        // pos = contact_pos_in[contactid]                                                        <L 1729>
        var_28 = wp::address(var_contact_pos_in, var_0);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // if id1:                                                                                <L 1732>
        if (var_11) {
            // com1 = subtree_com_in[worldid, body_rootid[id1]]                                   <L 1733>
            var_31 = wp::address(var_body_rootid, var_11);
            var_33 = wp::load(var_31);
            var_32 = wp::address(var_subtree_com_in, var_24, var_33);
            var_35 = wp::load(var_32);
            var_34 = wp::copy(var_35);
            // wp.atomic_sub(cfrc_ext_out[worldid], id1, support.transform_force(force, com1 - pos))       <L 1734>
            var_36 = wp::view(var_cfrc_ext_out, var_24);
            var_37 = wp::sub(var_34, var_29);
            var_38 = transform_force_1(var_27, var_37);
            var_39 = wp::atomic_sub(var_36, var_11, var_38);
        }
        // if id2:                                                                                <L 1736>
        if (var_16) {
            // com2 = subtree_com_in[worldid, body_rootid[id2]]                                   <L 1737>
            var_40 = wp::address(var_body_rootid, var_16);
            var_42 = wp::load(var_40);
            var_41 = wp::address(var_subtree_com_in, var_24, var_42);
            var_44 = wp::load(var_41);
            var_43 = wp::copy(var_44);
            // wp.atomic_add(cfrc_ext_out[worldid], id2, support.transform_force(force, com2 - pos))       <L 1738>
            var_45 = wp::view(var_cfrc_ext_out, var_24);
            var_46 = wp::sub(var_43, var_29);
            var_47 = transform_force_1(var_27, var_46);
            var_48 = wp::atomic_add(var_45, var_16, var_47);
        }
    }
}



extern "C" __global__ void _transmission_body_moment_af66f321_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_actuator_trntype_body_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::int32> var_nacon_in,
    bool var_efc_is_sparse,
    wp::array_t<wp::float32> var_actuator_moment_out,
    wp::array_t<wp::int32> var_actuator_trntype_body_ncon_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        const wp::int32 var_10 = 0;
        wp::int32* var_11;
        bool var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        bool var_36;
        bool var_37;
        wp::float32* var_38;
        wp::float32* var_39;
        bool var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        wp::array_t<wp::int32> var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::array_t<wp::int32> var_57;
        bool var_58;
        const wp::int32 var_59 = 1;
        bool var_60;
        const wp::int32 var_61 = 1;
        bool var_62;
        const wp::int32 var_63 = 0;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        bool var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        const wp::int32 var_75 = 0;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::array_t<wp::float32> var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::int32 var_85;
        wp::array_t<wp::float32> var_86;
        wp::int32 var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        const wp::float32 var_94 = 0.5;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        wp::range_t var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        bool var_107;
        wp::int32* var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        const wp::int32 var_112 = 0;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::array_t<wp::float32> var_116;
        wp::int32 var_117;
        const wp::int32 var_118 = 0;
        wp::float32* var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::int32 var_123;
        wp::array_t<wp::float32> var_124;
        wp::int32 var_125;
        wp::float32* var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        const wp::int32 var_135 = 1;
        bool var_136;
        wp::vec_t<3, wp::float32>* var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::mat_t<3, 3, wp::float32>* var_140;
        wp::mat_t<3, 3, wp::float32> var_141;
        wp::mat_t<3, 3, wp::float32> var_142;
        const wp::int32 var_143 = 0;
        const wp::int32 var_144 = 0;
        wp::float32 var_145;
        const wp::int32 var_146 = 0;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        const wp::int32 var_149 = 0;
        const wp::int32 var_150 = 2;
        wp::float32 var_151;
        wp::vec_t<3, wp::float32> var_152;
        const wp::int32 var_153 = 0;
        wp::int32* var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        bool var_157;
        const wp::int32 var_158 = 0;
        bool var_159;
        wp::int32* var_160;
        bool var_161;
        wp::int32 var_162;
        wp::int32* var_163;
        wp::int32 var_164;
        wp::int32 var_165;
        const wp::int32 var_166 = 0;
        wp::int32* var_167;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::vec_t<3, wp::float32> var_174;
        wp::vec_t<3, wp::float32> var_175;
        wp::vec_t<3, wp::float32> var_176;
        wp::array_t<wp::float32> var_177;
        wp::int32 var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        //---------
        // forward
        // def _transmission_body_moment(                                                         <L 2730>
        // trnbodyid, conid, dofid = wp.tid()                                                     <L 2764>
        builtin_tid3d(var_0, var_1, var_2);
        // actid = actuator_trntype_body_adr[trnbodyid]                                           <L 2765>
        var_3 = wp::address(var_actuator_trntype_body_adr, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // bodyid = actuator_trnid[actid][0]                                                      <L 2766>
        var_6 = wp::address(var_actuator_trnid, var_4);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        // if conid >= nacon_in[0]:                                                               <L 2768>
        var_11 = wp::address(var_nacon_in, var_10);
        var_13 = wp::load(var_11);
        var_12 = (var_1 >= var_13);
        if (var_12) {
            // return                                                                             <L 2769>
            continue;
        }
        // worldid = contact_worldid_in[conid]                                                    <L 2771>
        var_14 = wp::address(var_contact_worldid_in, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // geom = contact_geom_in[conid]                                                          <L 2774>
        var_17 = wp::address(var_contact_geom_in, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // g1 = geom[0]                                                                           <L 2775>
        var_21 = wp::extract(var_18, var_20);
        // g2 = geom[1]                                                                           <L 2776>
        var_23 = wp::extract(var_18, var_22);
        // if g1 < 0 or g2 < 0:                                                                   <L 2779>
        var_26 = (var_21 < var_25);
        var_24 = var_26;
        if (!var_24) {
            var_28 = (var_23 < var_27);
            var_24 = var_24 || var_28;
        }
        if (var_24) {
            // return                                                                             <L 2780>
            continue;
        }
        // b1 = geom_bodyid[g1]                                                                   <L 2783>
        var_29 = wp::address(var_geom_bodyid, var_21);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // b2 = geom_bodyid[g2]                                                                   <L 2784>
        var_32 = wp::address(var_geom_bodyid, var_23);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // if b1 != bodyid and b2 != bodyid:                                                      <L 2787>
        var_36 = (var_30 != var_8);
        var_35 = var_36;
        if (var_35) {
            var_37 = (var_33 != var_8);
            var_35 = var_35 && var_37;
        }
        if (var_35) {
            // return                                                                             <L 2788>
            continue;
        }
        // contact_exclude = int(contact_dist_in[conid] >= contact_includemargin_in[conid])       <L 2790>
        var_38 = wp::address(var_contact_dist_in, var_1);
        var_39 = wp::address(var_contact_includemargin_in, var_1);
        var_41 = wp::load(var_38);
        var_42 = wp::load(var_39);
        var_40 = (var_41 >= var_42);
        var_43 = wp::int(var_40);
        // if dofid == 0:                                                                         <L 2792>
        var_45 = (var_2 == var_44);
        if (var_45) {
            // wp.atomic_add(actuator_trntype_body_ncon_out[worldid], trnbodyid, 1)               <L 2793>
            var_46 = wp::view(var_actuator_trntype_body_ncon_out, var_15);
            var_48 = wp::atomic_add(var_46, var_0, var_47);
        }
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 2795>
        var_49 = wp::address(var_moment_rowadr_in, var_15, var_4);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // if contact_exclude == 0:                                                               <L 2798>
        var_53 = (var_43 == var_52);
        if (var_53) {
            // contact_dim = contact_dim_in[conid]                                                <L 2799>
            var_54 = wp::address(var_contact_dim_in, var_1);
            var_56 = wp::load(var_54);
            var_55 = wp::copy(var_56);
            // contact_efc_address = contact_efc_address_in[conid]                                <L 2800>
            var_57 = wp::view(var_contact_efc_address_in, var_1);
            // if contact_dim == 1 or opt_cone == ConeType.ELLIPTIC:                              <L 2802>
            var_60 = (var_55 == var_59);
            var_58 = var_60;
            if (!var_58) {
                var_62 = (var_opt_cone == var_61);
                var_58 = var_58 || var_62;
            }
            if (var_58) {
                // efcid0 = contact_efc_address[0]                                                <L 2803>
                var_64 = wp::address(var_57, var_63);
                var_66 = wp::load(var_64);
                var_65 = wp::copy(var_66);
                // if efc_is_sparse:                                                              <L 2804>
                if (var_efc_is_sparse) {
                    // rownnz = efc_J_rownnz_in[worldid, efcid0]                                  <L 2805>
                    var_67 = wp::address(var_efc_J_rownnz_in, var_15, var_65);
                    var_69 = wp::load(var_67);
                    var_68 = wp::copy(var_69);
                    // if dofid < rownnz:                                                         <L 2806>
                    var_70 = (var_2 < var_68);
                    if (var_70) {
                        // efc_rowadr = efc_J_rowadr_in[worldid, efcid0]                          <L 2807>
                        var_71 = wp::address(var_efc_J_rowadr_in, var_15, var_65);
                        var_73 = wp::load(var_71);
                        var_72 = wp::copy(var_73);
                        // efc_sparseid = efc_rowadr + dofid                                      <L 2808>
                        var_74 = wp::add(var_72, var_2);
                        // colind = efc_J_colind_in[worldid, 0, efc_sparseid]                     <L 2809>
                        var_76 = wp::address(var_efc_J_colind_in, var_15, var_75, var_74);
                        var_78 = wp::load(var_76);
                        var_77 = wp::copy(var_78);
                        // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid])       <L 2810>
                        var_79 = wp::view(var_actuator_moment_out, var_15);
                        var_80 = wp::add(var_50, var_77);
                        var_82 = wp::address(var_efc_J_in, var_15, var_81, var_74);
                        var_84 = wp::load(var_82);
                        var_83 = wp::atomic_add(var_79, var_80, var_84);
                    }
                    if (!var_70) {
                        // return                                                                 <L 2812>
                        continue;
                    }
                }
                if (!var_efc_is_sparse) {
                    // colind = dofid                                                             <L 2814>
                    var_85 = wp::copy(var_2);
                    // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid0, dofid])       <L 2815>
                    var_86 = wp::view(var_actuator_moment_out, var_15);
                    var_87 = wp::add(var_50, var_85);
                    var_88 = wp::address(var_efc_J_in, var_15, var_65, var_2);
                    var_90 = wp::load(var_88);
                    var_89 = wp::atomic_add(var_86, var_87, var_90);
                }
                var_91 = wp::where(var_efc_is_sparse, var_77, var_85);
            }
            if (!var_58) {
                // npyramid = contact_dim - 1  # number of frictional directions                  <L 2817>
                var_93 = wp::sub(var_55, var_92);
                // efc_force = 0.5 / float(npyramid)                                              <L 2818>
                var_95 = wp::float(var_93);
                var_96 = wp::div(var_94, var_95);
                // for j in range(2 * npyramid):                                                  <L 2820>
                var_98 = wp::mul(var_97, var_93);
                var_99 = wp::range(var_98);
                start_for_4:;
                    if (iter_cmp(var_99) == 0) goto end_for_4;
                    var_100 = wp::iter_next(var_99);
                    // efcid = contact_efc_address[j]                                             <L 2821>
                    var_101 = wp::address(var_57, var_100);
                    var_103 = wp::load(var_101);
                    var_102 = wp::copy(var_103);
                    // if efc_is_sparse:                                                          <L 2822>
                    if (var_efc_is_sparse) {
                        // rownnz = efc_J_rownnz_in[worldid, efcid]                               <L 2823>
                        var_104 = wp::address(var_efc_J_rownnz_in, var_15, var_102);
                        var_106 = wp::load(var_104);
                        var_105 = wp::copy(var_106);
                        // if dofid < rownnz:                                                     <L 2824>
                        var_107 = (var_2 < var_105);
                        if (var_107) {
                            // efc_rowadr = efc_J_rowadr_in[worldid, efcid]                       <L 2825>
                            var_108 = wp::address(var_efc_J_rowadr_in, var_15, var_102);
                            var_110 = wp::load(var_108);
                            var_109 = wp::copy(var_110);
                            // efc_sparseid = efc_rowadr + dofid                                  <L 2826>
                            var_111 = wp::add(var_109, var_2);
                            // colind = efc_J_colind_in[worldid, 0, efc_sparseid]                 <L 2827>
                            var_113 = wp::address(var_efc_J_colind_in, var_15, var_112, var_111);
                            var_115 = wp::load(var_113);
                            var_114 = wp::copy(var_115);
                            // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid] * efc_force)       <L 2828>
                            var_116 = wp::view(var_actuator_moment_out, var_15);
                            var_117 = wp::add(var_50, var_114);
                            var_119 = wp::address(var_efc_J_in, var_15, var_118, var_111);
                            var_121 = wp::load(var_119);
                            var_120 = wp::mul(var_121, var_96);
                            var_122 = wp::atomic_add(var_116, var_117, var_120);
                        }
                        if (!var_107) {
                            // return                                                             <L 2830>
                            continue;
                        }
                    }
                    if (!var_efc_is_sparse) {
                        // colind = dofid                                                         <L 2832>
                        var_123 = wp::copy(var_2);
                        // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid, dofid] * efc_force)       <L 2833>
                        var_124 = wp::view(var_actuator_moment_out, var_15);
                        var_125 = wp::add(var_50, var_123);
                        var_126 = wp::address(var_efc_J_in, var_15, var_102, var_2);
                        var_128 = wp::load(var_126);
                        var_127 = wp::mul(var_128, var_96);
                        var_129 = wp::atomic_add(var_124, var_125, var_127);
                    }
                    var_130 = wp::where(var_efc_is_sparse, var_114, var_123);
                    goto start_for_4;
                end_for_4:;
            }
            var_131 = wp::where(var_58, var_68, var_105);
            var_132 = wp::where(var_58, var_72, var_109);
            var_133 = wp::where(var_58, var_74, var_111);
            var_134 = wp::where(var_58, var_91, var_130);
        }
        if (!var_53) {
            // elif contact_exclude == 1:                                                         <L 2836>
            var_136 = (var_43 == var_135);
            if (var_136) {
                // contact_pos = contact_pos_in[conid]                                            <L 2837>
                var_137 = wp::address(var_contact_pos_in, var_1);
                var_139 = wp::load(var_137);
                var_138 = wp::copy(var_139);
                // contact_frame = contact_frame_in[conid]                                        <L 2838>
                var_140 = wp::address(var_contact_frame_in, var_1);
                var_142 = wp::load(var_140);
                var_141 = wp::copy(var_142);
                // normal = wp.vec3(contact_frame[0, 0], contact_frame[0, 1], contact_frame[0, 2])       <L 2839>
                var_145 = wp::extract(var_141, var_143, var_144);
                var_148 = wp::extract(var_141, var_146, var_147);
                var_151 = wp::extract(var_141, var_149, var_150);
                var_152 = wp::vec_t<3, wp::float32>(var_145, var_148, var_151);
                // efcid0 = contact_efc_address_in[conid][0]                                      <L 2842>
                var_154 = wp::address(var_contact_efc_address_in, var_1, var_153);
                var_156 = wp::load(var_154);
                var_155 = wp::copy(var_156);
                // if efc_is_sparse and efcid0 >= 0:                                              <L 2843>
                var_157 = var_efc_is_sparse;
                if (var_157) {
                    var_159 = (var_155 >= var_158);
                    var_157 = var_157 && var_159;
                }
                if (var_157) {
                    // if dofid >= efc_J_rownnz_in[worldid, efcid0]:                              <L 2845>
                    var_160 = wp::address(var_efc_J_rownnz_in, var_15, var_155);
                    var_162 = wp::load(var_160);
                    var_161 = (var_2 >= var_162);
                    if (var_161) {
                        // return                                                                 <L 2846>
                        continue;
                    }
                    // sparseid = efc_J_rowadr_in[worldid, efcid0] + dofid                        <L 2847>
                    var_163 = wp::address(var_efc_J_rowadr_in, var_15, var_155);
                    var_165 = wp::load(var_163);
                    var_164 = wp::add(var_165, var_2);
                    // colind = efc_J_colind_in[worldid, 0, sparseid]                             <L 2848>
                    var_167 = wp::address(var_efc_J_colind_in, var_15, var_166, var_164);
                    var_169 = wp::load(var_167);
                    var_168 = wp::copy(var_169);
                }
                if (!var_157) {
                    // colind = dofid                                                             <L 2851>
                    var_170 = wp::copy(var_2);
                }
                var_171 = wp::where(var_157, var_168, var_170);
                // jacp1, _ = support.jac_dof(                                                    <L 2853>
                // body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, contact_pos, b1, colind, worldid       <L 2854>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_138, var_30, var_171, var_15, var_172, var_173);
                // jacp2, _ = support.jac_dof(                                                    <L 2856>
                // body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, contact_pos, b2, colind, worldid       <L 2857>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_body_isdofancestor, var_subtree_com_in, var_cdof_in, var_138, var_33, var_171, var_15, var_174, var_175);
                // jacdif = jacp2 - jacp1                                                         <L 2860>
                var_176 = wp::sub(var_174, var_172);
                // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, wp.dot(normal, jacdif))       <L 2863>
                var_177 = wp::view(var_actuator_moment_out, var_15);
                var_178 = wp::add(var_50, var_171);
                var_179 = wp::dot(var_152, var_176);
                var_180 = wp::atomic_add(var_177, var_178, var_179);
            }
        }
        var_181 = wp::where(var_53, var_65, var_155);
        var_182 = wp::where(var_53, var_134, var_171);
    }
}



extern "C" __global__ void _compute_body_inertial_frames_d3bdd81a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_ipos,
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32>* var_5;
        wp::quat_t<wp::float32> var_6;
        wp::quat_t<wp::float32> var_7;
        wp::shape_t* var_8;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        wp::shape_t var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::quat_t<wp::float32>* var_22;
        wp::quat_t<wp::float32> var_23;
        wp::quat_t<wp::float32> var_24;
        wp::mat_t<3, 3, wp::float32> var_25;
        //---------
        // forward
        // def _compute_body_inertial_frames(                                                     <L 149>
        // worldid, bodyid = wp.tid()                                                             <L 160>
        builtin_tid2d(var_0, var_1);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 161>
        var_2 = wp::address(var_xpos_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 162>
        var_5 = wp::address(var_xquat_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // xipos_out[worldid, bodyid] = xpos + math.rot_vec_quat(body_ipos[worldid % body_ipos.shape[0], bodyid], xquat)       <L 163>
        var_8 = &(var_body_ipos.shape);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        var_12 = wp::mod(var_0, var_10);
        var_13 = wp::address(var_body_ipos, var_12, var_1);
        var_15 = wp::load(var_13);
        var_14 = rot_vec_quat_0(var_15, var_6);
        var_16 = wp::add(var_3, var_14);
        wp::array_store(var_xipos_out, var_0, var_1, var_16);
        // ximat_out[worldid, bodyid] = math.quat_to_mat(math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], bodyid]))       <L 164>
        var_17 = &(var_body_iquat.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_21 = wp::mod(var_0, var_19);
        var_22 = wp::address(var_body_iquat, var_21, var_1);
        var_24 = wp::load(var_22);
        var_23 = mul_quat_0(var_6, var_24);
        var_25 = quat_to_mat_0(var_23);
        wp::array_store(var_ximat_out, var_0, var_1, var_25);
    }
}



extern "C" __global__ void _spatial_site_tendon_d90d9688_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::int32> var_tendon_site_pair_adr,
    wp::array_t<wp::int32> var_wrap_site_pair_adr,
    wp::array_t<wp::float32> var_wrap_pulley_scale,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::float32 var_29;
        wp::array_t<wp::float32> var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::float32 var_33 = 1e-15;
        bool var_34;
        const wp::float32 var_35 = 1.0;
        const wp::float32 var_36 = 0.0;
        const wp::float32 var_37 = 0.0;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        bool var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::int32 var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::int32* var_58;
        wp::vec_t<3, wp::float32>* var_59;
        wp::int32 var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::float32 var_63;
        //---------
        // forward
        // def _spatial_site_tendon(                                                              <L 3740>
        // worldid, elementid = wp.tid()                                                          <L 3762>
        builtin_tid2d(var_0, var_1);
        // site_pair_adr = wrap_site_pair_adr[elementid]                                          <L 3765>
        var_2 = wp::address(var_wrap_site_pair_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tenid = tendon_site_pair_adr[elementid]                                                <L 3766>
        var_5 = wp::address(var_tendon_site_pair_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // pulley_scale = wrap_pulley_scale[site_pair_adr]                                        <L 3769>
        var_8 = wp::address(var_wrap_pulley_scale, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // id0 = wrap_objid[site_pair_adr + 0]                                                    <L 3771>
        var_12 = wp::add(var_3, var_11);
        var_13 = wp::address(var_wrap_objid, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // id1 = wrap_objid[site_pair_adr + 1]                                                    <L 3772>
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::address(var_wrap_objid, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // pnt0 = site_xpos_in[worldid, id0]                                                      <L 3774>
        var_21 = wp::address(var_site_xpos_in, var_0, var_14);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // pnt1 = site_xpos_in[worldid, id1]                                                      <L 3775>
        var_24 = wp::address(var_site_xpos_in, var_0, var_19);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // dif = pnt1 - pnt0                                                                      <L 3776>
        var_27 = wp::sub(var_25, var_22);
        // vec, length = math.normalize_with_norm(dif)                                            <L 3777>
        normalize_with_norm_0(var_27, var_28, var_29);
        // wp.atomic_add(ten_length_out[worldid], tenid, length * pulley_scale)                   <L 3778>
        var_30 = wp::view(var_ten_length_out, var_0);
        var_31 = wp::mul(var_29, var_9);
        var_32 = wp::atomic_add(var_30, var_6, var_31);
        // if length < MJ_MINVAL:                                                                 <L 3780>
        var_34 = (var_29 < var_33);
        if (var_34) {
            // vec = wp.vec3(1.0, 0.0, 0.0)                                                       <L 3781>
            var_38 = wp::vec_t<3, wp::float32>(var_35, var_36, var_37);
        }
        var_39 = wp::where(var_34, var_38, var_28);
        // body0 = site_bodyid[id0]                                                               <L 3783>
        var_40 = wp::address(var_site_bodyid, var_14);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // body1 = site_bodyid[id1]                                                               <L 3784>
        var_43 = wp::address(var_site_bodyid, var_19);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // if body0 != body1:                                                                     <L 3785>
        var_46 = (var_41 != var_44);
        if (var_46) {
            // rownnz = ten_J_rownnz[tenid]                                                       <L 3786>
            var_47 = wp::address(var_ten_J_rownnz, var_6);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 3787>
            var_50 = wp::address(var_ten_J_rowadr, var_6);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // offset0 = pnt0 - subtree_com_in[worldid, body_rootid[body0]]                       <L 3788>
            var_53 = wp::address(var_body_rootid, var_41);
            var_55 = wp::load(var_53);
            var_54 = wp::address(var_subtree_com_in, var_0, var_55);
            var_57 = wp::load(var_54);
            var_56 = wp::sub(var_22, var_57);
            // offset1 = pnt1 - subtree_com_in[worldid, body_rootid[body1]]                       <L 3789>
            var_58 = wp::address(var_body_rootid, var_44);
            var_60 = wp::load(var_58);
            var_59 = wp::address(var_subtree_com_in, var_0, var_60);
            var_62 = wp::load(var_59);
            var_61 = wp::sub(var_25, var_62);
            // _accumulate_jac_chain(                                                             <L 3790>
            // body_parentid,                                                                     <L 3791>
            // body_dofnum,                                                                       <L 3792>
            // body_dofadr,                                                                       <L 3793>
            // ten_J_colind,                                                                      <L 3794>
            // cdof_in,                                                                           <L 3795>
            // offset0,                                                                           <L 3796>
            // vec,                                                                               <L 3797>
            // body0,                                                                             <L 3798>
            // rowadr,                                                                            <L 3799>
            // rownnz,                                                                            <L 3800>
            // -pulley_scale,                                                                     <L 3801>
            var_63 = wp::neg(var_9);
            // worldid,                                                                           <L 3802>
            // ten_J_out,                                                                         <L 3803>
            _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_56, var_39, var_41, var_51, var_48, var_63, var_0, var_ten_J_out);
            // _accumulate_jac_chain(                                                             <L 3805>
            // body_parentid,                                                                     <L 3806>
            // body_dofnum,                                                                       <L 3807>
            // body_dofadr,                                                                       <L 3808>
            // ten_J_colind,                                                                      <L 3809>
            // cdof_in,                                                                           <L 3810>
            // offset1,                                                                           <L 3811>
            // vec,                                                                               <L 3812>
            // body1,                                                                             <L 3813>
            // rowadr,                                                                            <L 3814>
            // rownnz,                                                                            <L 3815>
            // pulley_scale,                                                                      <L 3816>
            // worldid,                                                                           <L 3817>
            // ten_J_out,                                                                         <L 3818>
            _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_61, var_39, var_44, var_51, var_48, var_9, var_0, var_ten_J_out);
        }
    }
}



extern "C" __global__ void _transmission_body_moment_scale_e0bec4b0_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_actuator_trntype_body_adr,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_actuator_trntype_body_ncon_in,
    wp::array_t<wp::float32> var_actuator_moment_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def _transmission_body_moment_scale(                                                   <L 2867>
        // worldid, trnbodyid, dofid = wp.tid()                                                   <L 2877>
        builtin_tid3d(var_0, var_1, var_2);
        // ncon = actuator_trntype_body_ncon_in[worldid, trnbodyid]                               <L 2879>
        var_3 = wp::address(var_actuator_trntype_body_ncon_in, var_0, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if ncon > 0:                                                                           <L 2881>
        var_7 = (var_4 > var_6);
        if (var_7) {
            // actid = actuator_trntype_body_adr[trnbodyid]                                       <L 2882>
            var_8 = wp::address(var_actuator_trntype_body_adr, var_1);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // rowadr = moment_rowadr_in[worldid, actid]                                          <L 2883>
            var_11 = wp::address(var_moment_rowadr_in, var_0, var_9);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // actuator_moment_out[worldid, rowadr + dofid] /= -float(ncon)                       <L 2884>
            var_14 = wp::float(var_4);
            var_15 = wp::neg(var_14);
            var_16 = wp::add(var_12, var_2);
            var_17 = wp::address(var_actuator_moment_out, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::div(var_19, var_15);
            wp::array_store(var_actuator_moment_out, var_0, var_16, var_18);
        }
    }
}



extern "C" __global__ void _geom_local_to_global_e28b714c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = -1;
        bool var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::quat_t<wp::float32>* var_19;
        wp::quat_t<wp::float32> var_20;
        wp::quat_t<wp::float32> var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::shape_t* var_31;
        const wp::int32 var_32 = 0;
        wp::int32 var_33;
        wp::shape_t var_34;
        wp::int32 var_35;
        wp::quat_t<wp::float32>* var_36;
        wp::quat_t<wp::float32> var_37;
        wp::quat_t<wp::float32> var_38;
        wp::mat_t<3, 3, wp::float32> var_39;
        //---------
        // forward
        // def _geom_local_to_global(                                                             <L 179>
        // worldid, geomid = wp.tid()                                                             <L 194>
        builtin_tid2d(var_0, var_1);
        // bodyid = geom_bodyid[geomid]                                                           <L 195>
        var_2 = wp::address(var_geom_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if body_weldid[bodyid] == 0 and body_mocapid[body_rootid[bodyid]] == -1:               <L 197>
        var_6 = wp::address(var_body_weldid, var_3);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        var_5 = var_8;
        if (var_5) {
            var_10 = wp::address(var_body_rootid, var_3);
            var_12 = wp::load(var_10);
            var_11 = wp::address(var_body_mocapid, var_12);
            var_15 = wp::load(var_11);
            var_14 = (var_15 == var_13);
            var_5 = var_5 && var_14;
        }
        if (var_5) {
            // return                                                                             <L 200>
            continue;
        }
        // xpos = xpos_in[worldid, bodyid]                                                        <L 202>
        var_16 = wp::address(var_xpos_in, var_0, var_3);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 203>
        var_19 = wp::address(var_xquat_in, var_0, var_3);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // geom_xpos_out[worldid, geomid] = xpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], xquat)       <L 204>
        var_22 = &(var_geom_pos.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_0, var_24);
        var_27 = wp::address(var_geom_pos, var_26, var_1);
        var_29 = wp::load(var_27);
        var_28 = rot_vec_quat_0(var_29, var_20);
        var_30 = wp::add(var_17, var_28);
        wp::array_store(var_geom_xpos_out, var_0, var_1, var_30);
        // geom_xmat_out[worldid, geomid] = math.quat_to_mat(math.mul_quat(xquat, geom_quat[worldid % geom_quat.shape[0], geomid]))       <L 205>
        var_31 = &(var_geom_quat.shape);
        var_34 = wp::load(var_31);
        var_33 = wp::extract(var_34, var_32);
        var_35 = wp::mod(var_0, var_33);
        var_36 = wp::address(var_geom_quat, var_35, var_1);
        var_38 = wp::load(var_36);
        var_37 = mul_quat_0(var_20, var_38);
        var_39 = quat_to_mat_0(var_37);
        wp::array_store(var_geom_xmat_out, var_0, var_1, var_39);
    }
}



extern "C" __global__ void _cam_local_to_global_ed67ebfa_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cam_mode,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_poscom0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos0,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_mat0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32* var_13;
        const wp::int32 var_14 = 3;
        bool var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 4;
        bool var_19;
        wp::int32 var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::quat_t<wp::float32>* var_32;
        wp::quat_t<wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::quat_t<wp::float32>* var_39;
        wp::quat_t<wp::float32> var_40;
        wp::quat_t<wp::float32> var_41;
        wp::mat_t<3, 3, wp::float32> var_42;
        wp::int32* var_43;
        const wp::int32 var_44 = 1;
        bool var_45;
        wp::int32 var_46;
        wp::shape_t* var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        wp::shape_t var_50;
        wp::int32 var_51;
        wp::mat_t<3, 3, wp::float32>* var_52;
        wp::mat_t<3, 3, wp::float32> var_53;
        wp::int32* var_54;
        wp::vec_t<3, wp::float32>* var_55;
        wp::int32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::shape_t* var_59;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        wp::shape_t var_62;
        wp::int32 var_63;
        wp::vec_t<3, wp::float32>* var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::int32* var_67;
        const wp::int32 var_68 = 2;
        bool var_69;
        wp::int32 var_70;
        wp::shape_t* var_71;
        const wp::int32 var_72 = 0;
        wp::int32 var_73;
        wp::shape_t var_74;
        wp::int32 var_75;
        wp::mat_t<3, 3, wp::float32>* var_76;
        wp::mat_t<3, 3, wp::float32> var_77;
        wp::int32* var_78;
        wp::vec_t<3, wp::float32>* var_79;
        wp::int32 var_80;
        wp::shape_t* var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        wp::shape_t var_84;
        wp::int32 var_85;
        wp::vec_t<3, wp::float32>* var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        bool var_90;
        wp::int32* var_91;
        const wp::int32 var_92 = 3;
        bool var_93;
        wp::int32 var_94;
        wp::int32* var_95;
        const wp::int32 var_96 = 4;
        bool var_97;
        wp::int32 var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        wp::vec_t<3, wp::float32>* var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::quat_t<wp::float32>* var_105;
        wp::quat_t<wp::float32> var_106;
        wp::quat_t<wp::float32> var_107;
        wp::vec_t<3, wp::float32>* var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::int32* var_112;
        wp::vec_t<3, wp::float32>* var_113;
        wp::int32 var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::int32* var_117;
        const wp::int32 var_118 = 4;
        bool var_119;
        wp::int32 var_120;
        wp::int32* var_121;
        wp::vec_t<3, wp::float32>* var_122;
        wp::int32 var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32>* var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        const wp::float32 var_131 = 0.0;
        const wp::float32 var_132 = 0.0;
        const wp::float32 var_133 = 1.0;
        wp::vec_t<3, wp::float32> var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        const wp::int32 var_139 = 0;
        wp::float32 var_140;
        const wp::int32 var_141 = 0;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::float32 var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        const wp::int32 var_149 = 1;
        wp::float32 var_150;
        const wp::int32 var_151 = 2;
        wp::float32 var_152;
        const wp::int32 var_153 = 2;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        wp::mat_t<3, 3, wp::float32> var_157;
        wp::int32* var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        wp::vec_t<3, wp::float32>* var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::vec_t<3, wp::float32> var_163;
        wp::quat_t<wp::float32>* var_164;
        wp::quat_t<wp::float32> var_165;
        wp::quat_t<wp::float32> var_166;
        wp::vec_t<3, wp::float32>* var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::vec_t<3, wp::float32> var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::quat_t<wp::float32>* var_171;
        wp::quat_t<wp::float32> var_172;
        wp::quat_t<wp::float32> var_173;
        wp::mat_t<3, 3, wp::float32> var_174;
        wp::int32 var_175;
        wp::vec_t<3, wp::float32> var_176;
        wp::quat_t<wp::float32> var_177;
        wp::int32 var_178;
        wp::vec_t<3, wp::float32> var_179;
        wp::quat_t<wp::float32> var_180;
        //---------
        // forward
        // def _cam_local_to_global(                                                              <L 859>
        // worldid, camid = wp.tid()                                                              <L 877>
        builtin_tid2d(var_0, var_1);
        // cam_pos_id = worldid % cam_pos.shape[0]                                                <L 878>
        var_2 = &(var_cam_pos.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // cam_quat_id = worldid % cam_quat.shape[0]                                              <L 879>
        var_7 = &(var_cam_quat.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // is_target_cam = (cam_mode[camid] == CamLightType.TARGETBODY) or (cam_mode[camid] == CamLightType.TARGETBODYCOM)       <L 880>
        var_13 = wp::address(var_cam_mode, var_1);
        var_16 = wp::load(var_13);
        var_15 = (var_16 == var_14);
        var_12 = var_15;
        if (!var_12) {
            var_17 = wp::address(var_cam_mode, var_1);
            var_20 = wp::load(var_17);
            var_19 = (var_20 == var_18);
            var_12 = var_12 || var_19;
        }
        // invalid_target = is_target_cam and (cam_targetbodyid[camid] < 0)                       <L 881>
        var_21 = var_12;
        if (var_21) {
            var_22 = wp::address(var_cam_targetbodyid, var_1);
            var_25 = wp::load(var_22);
            var_24 = (var_25 < var_23);
            var_21 = var_21 && var_24;
        }
        // if invalid_target:                                                                     <L 882>
        if (var_21) {
            // bodyid = cam_bodyid[camid]                                                         <L 883>
            var_26 = wp::address(var_cam_bodyid, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 884>
            var_29 = wp::address(var_xpos_in, var_0, var_27);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xquat = xquat_in[worldid, bodyid]                                                  <L 885>
            var_32 = wp::address(var_xquat_in, var_0, var_27);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 886>
            var_35 = wp::address(var_cam_pos, var_6, var_1);
            var_37 = wp::load(var_35);
            var_36 = rot_vec_quat_0(var_37, var_33);
            var_38 = wp::add(var_30, var_36);
            wp::array_store(var_cam_xpos_out, var_0, var_1, var_38);
            // cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))       <L 887>
            var_39 = wp::address(var_cam_quat, var_11, var_1);
            var_41 = wp::load(var_39);
            var_40 = mul_quat_0(var_33, var_41);
            var_42 = quat_to_mat_0(var_40);
            wp::array_store(var_cam_xmat_out, var_0, var_1, var_42);
        }
        if (!var_21) {
            // elif cam_mode[camid] == CamLightType.TRACK:                                        <L 888>
            var_43 = wp::address(var_cam_mode, var_1);
            var_46 = wp::load(var_43);
            var_45 = (var_46 == var_44);
            if (var_45) {
                // cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]       <L 889>
                var_47 = &(var_cam_mat0.shape);
                var_50 = wp::load(var_47);
                var_49 = wp::extract(var_50, var_48);
                var_51 = wp::mod(var_0, var_49);
                var_52 = wp::address(var_cam_mat0, var_51, var_1);
                var_53 = wp::load(var_52);
                wp::array_store(var_cam_xmat_out, var_0, var_1, var_53);
                // body_xpos = xpos_in[worldid, cam_bodyid[camid]]                                <L 890>
                var_54 = wp::address(var_cam_bodyid, var_1);
                var_56 = wp::load(var_54);
                var_55 = wp::address(var_xpos_in, var_0, var_56);
                var_58 = wp::load(var_55);
                var_57 = wp::copy(var_58);
                // cam_xpos_out[worldid, camid] = body_xpos + cam_pos0[worldid % cam_pos0.shape[0], camid]       <L 891>
                var_59 = &(var_cam_pos0.shape);
                var_62 = wp::load(var_59);
                var_61 = wp::extract(var_62, var_60);
                var_63 = wp::mod(var_0, var_61);
                var_64 = wp::address(var_cam_pos0, var_63, var_1);
                var_66 = wp::load(var_64);
                var_65 = wp::add(var_57, var_66);
                wp::array_store(var_cam_xpos_out, var_0, var_1, var_65);
            }
            if (!var_45) {
                // elif cam_mode[camid] == CamLightType.TRACKCOM:                                 <L 892>
                var_67 = wp::address(var_cam_mode, var_1);
                var_70 = wp::load(var_67);
                var_69 = (var_70 == var_68);
                if (var_69) {
                    // cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]       <L 893>
                    var_71 = &(var_cam_mat0.shape);
                    var_74 = wp::load(var_71);
                    var_73 = wp::extract(var_74, var_72);
                    var_75 = wp::mod(var_0, var_73);
                    var_76 = wp::address(var_cam_mat0, var_75, var_1);
                    var_77 = wp::load(var_76);
                    wp::array_store(var_cam_xmat_out, var_0, var_1, var_77);
                    // cam_xpos_out[worldid, camid] = (                                           <L 894>
                    // subtree_com_in[worldid, cam_bodyid[camid]] + cam_poscom0[worldid % cam_poscom0.shape[0], camid]       <L 895>
                    var_78 = wp::address(var_cam_bodyid, var_1);
                    var_80 = wp::load(var_78);
                    var_79 = wp::address(var_subtree_com_in, var_0, var_80);
                    var_81 = &(var_cam_poscom0.shape);
                    var_84 = wp::load(var_81);
                    var_83 = wp::extract(var_84, var_82);
                    var_85 = wp::mod(var_0, var_83);
                    var_86 = wp::address(var_cam_poscom0, var_85, var_1);
                    var_88 = wp::load(var_79);
                    var_89 = wp::load(var_86);
                    var_87 = wp::add(var_88, var_89);
                    // cam_xpos_out[worldid, camid] = (                                           <L 894>
                    wp::array_store(var_cam_xpos_out, var_0, var_1, var_87);
                }
                if (!var_69) {
                    // elif cam_mode[camid] == CamLightType.TARGETBODY or cam_mode[camid] == CamLightType.TARGETBODYCOM:       <L 897>
                    var_91 = wp::address(var_cam_mode, var_1);
                    var_94 = wp::load(var_91);
                    var_93 = (var_94 == var_92);
                    var_90 = var_93;
                    if (!var_90) {
                        var_95 = wp::address(var_cam_mode, var_1);
                        var_98 = wp::load(var_95);
                        var_97 = (var_98 == var_96);
                        var_90 = var_90 || var_97;
                    }
                    if (var_90) {
                        // bodyid = cam_bodyid[camid]                                             <L 898>
                        var_99 = wp::address(var_cam_bodyid, var_1);
                        var_101 = wp::load(var_99);
                        var_100 = wp::copy(var_101);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 899>
                        var_102 = wp::address(var_xpos_in, var_0, var_100);
                        var_104 = wp::load(var_102);
                        var_103 = wp::copy(var_104);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 900>
                        var_105 = wp::address(var_xquat_in, var_0, var_100);
                        var_107 = wp::load(var_105);
                        var_106 = wp::copy(var_107);
                        // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 901>
                        var_108 = wp::address(var_cam_pos, var_6, var_1);
                        var_110 = wp::load(var_108);
                        var_109 = rot_vec_quat_0(var_110, var_106);
                        var_111 = wp::add(var_103, var_109);
                        wp::array_store(var_cam_xpos_out, var_0, var_1, var_111);
                        // pos = xpos_in[worldid, cam_targetbodyid[camid]]                        <L 902>
                        var_112 = wp::address(var_cam_targetbodyid, var_1);
                        var_114 = wp::load(var_112);
                        var_113 = wp::address(var_xpos_in, var_0, var_114);
                        var_116 = wp::load(var_113);
                        var_115 = wp::copy(var_116);
                        // if cam_mode[camid] == CamLightType.TARGETBODYCOM:                      <L 903>
                        var_117 = wp::address(var_cam_mode, var_1);
                        var_120 = wp::load(var_117);
                        var_119 = (var_120 == var_118);
                        if (var_119) {
                            // pos = subtree_com_in[worldid, cam_targetbodyid[camid]]             <L 904>
                            var_121 = wp::address(var_cam_targetbodyid, var_1);
                            var_123 = wp::load(var_121);
                            var_122 = wp::address(var_subtree_com_in, var_0, var_123);
                            var_125 = wp::load(var_122);
                            var_124 = wp::copy(var_125);
                        }
                        var_126 = wp::where(var_119, var_124, var_115);
                        // mat_3 = wp.normalize(cam_xpos_out[worldid, camid] - pos)               <L 906>
                        var_127 = wp::address(var_cam_xpos_out, var_0, var_1);
                        var_129 = wp::load(var_127);
                        var_128 = wp::sub(var_129, var_126);
                        var_130 = wp::normalize(var_128);
                        // mat_1 = wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), mat_3))          <L 908>
                        var_134 = wp::vec_t<3, wp::float32>(var_131, var_132, var_133);
                        var_135 = wp::cross(var_134, var_130);
                        var_136 = wp::normalize(var_135);
                        // mat_2 = wp.normalize(wp.cross(mat_3, mat_1))                           <L 909>
                        var_137 = wp::cross(var_130, var_136);
                        var_138 = wp::normalize(var_137);
                        // cam_xmat_out[worldid, camid] = wp.mat33(                               <L 911>
                        // mat_1[0], mat_2[0], mat_3[0],                                          <L 912>
                        var_140 = wp::extract(var_136, var_139);
                        var_142 = wp::extract(var_138, var_141);
                        var_144 = wp::extract(var_130, var_143);
                        // mat_1[1], mat_2[1], mat_3[1],                                          <L 913>
                        var_146 = wp::extract(var_136, var_145);
                        var_148 = wp::extract(var_138, var_147);
                        var_150 = wp::extract(var_130, var_149);
                        // mat_1[2], mat_2[2], mat_3[2]                                           <L 914>
                        var_152 = wp::extract(var_136, var_151);
                        var_154 = wp::extract(var_138, var_153);
                        var_156 = wp::extract(var_130, var_155);
                        var_157 = wp::mat_t<3, 3, wp::float32>(var_140, var_142, var_144, var_146, var_148, var_150, var_152, var_154, var_156);
                        // cam_xmat_out[worldid, camid] = wp.mat33(                               <L 911>
                        wp::array_store(var_cam_xmat_out, var_0, var_1, var_157);
                    }
                    if (!var_90) {
                        // bodyid = cam_bodyid[camid]                                             <L 918>
                        var_158 = wp::address(var_cam_bodyid, var_1);
                        var_160 = wp::load(var_158);
                        var_159 = wp::copy(var_160);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 919>
                        var_161 = wp::address(var_xpos_in, var_0, var_159);
                        var_163 = wp::load(var_161);
                        var_162 = wp::copy(var_163);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 920>
                        var_164 = wp::address(var_xquat_in, var_0, var_159);
                        var_166 = wp::load(var_164);
                        var_165 = wp::copy(var_166);
                        // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 921>
                        var_167 = wp::address(var_cam_pos, var_6, var_1);
                        var_169 = wp::load(var_167);
                        var_168 = rot_vec_quat_0(var_169, var_165);
                        var_170 = wp::add(var_162, var_168);
                        wp::array_store(var_cam_xpos_out, var_0, var_1, var_170);
                        // cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))       <L 922>
                        var_171 = wp::address(var_cam_quat, var_11, var_1);
                        var_173 = wp::load(var_171);
                        var_172 = mul_quat_0(var_165, var_173);
                        var_174 = quat_to_mat_0(var_172);
                        wp::array_store(var_cam_xmat_out, var_0, var_1, var_174);
                    }
                    var_175 = wp::where(var_90, var_100, var_159);
                    var_176 = wp::where(var_90, var_103, var_162);
                    var_177 = wp::where(var_90, var_106, var_165);
                }
            }
        }
        var_178 = wp::where(var_21, var_27, var_175);
        var_179 = wp::where(var_21, var_30, var_176);
        var_180 = wp::where(var_21, var_33, var_177);
    }
}



extern "C" __global__ void _cfrc_ext_f67cbe30_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_xfrc_applied_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 0.0;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<6, wp::float32> var_10;
        const wp::int32 var_11 = 0;
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::int32* var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::int32 var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<6, wp::float32> var_24;
        //---------
        // forward
        // def _cfrc_ext(                                                                         <L 1519>
        // worldid, bodyid = wp.tid()                                                             <L 1529>
        builtin_tid2d(var_0, var_1);
        // if bodyid == 0:                                                                        <L 1530>
        var_3 = (var_1 == var_2);
        if (var_3) {
            // cfrc_ext_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 1531>
            var_10 = wp::vec_t<6, wp::float32>({var_4, var_5, var_6, var_7, var_8, var_9});
            wp::array_store(var_cfrc_ext_out, var_0, var_11, var_10);
        }
        if (!var_3) {
            // xfrc_applied = xfrc_applied_in[worldid, bodyid]                                    <L 1533>
            var_12 = wp::address(var_xfrc_applied_in, var_0, var_1);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]                         <L 1534>
            var_15 = wp::address(var_body_rootid, var_1);
            var_17 = wp::load(var_15);
            var_16 = wp::address(var_subtree_com_in, var_0, var_17);
            var_19 = wp::load(var_16);
            var_18 = wp::copy(var_19);
            // xipos = xipos_in[worldid, bodyid]                                                  <L 1535>
            var_20 = wp::address(var_xipos_in, var_0, var_1);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // cfrc_ext_out[worldid, bodyid] = support.transform_force(xfrc_applied, subtree_com - xipos)       <L 1536>
            var_23 = wp::sub(var_18, var_21);
            var_24 = transform_force_1(var_13, var_23);
            wp::array_store(var_cfrc_ext_out, var_0, var_1, var_24);
        }
    }
}



extern "C" __global__ void _cacc_branch_ffe1d6c4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    bool var_flg_acc,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<6, wp::float32>* var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32> var_18;
        wp::range_t var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::vec_t<6, wp::float32>* var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::vec_t<6, wp::float32> var_36;
        wp::vec_t<6, wp::float32> var_37;
        wp::float32 var_38;
        wp::vec_t<6, wp::float32> var_39;
        wp::int32 var_40;
        wp::vec_t<6, wp::float32>* var_41;
        wp::int32 var_42;
        wp::float32* var_43;
        wp::vec_t<6, wp::float32> var_44;
        wp::vec_t<6, wp::float32> var_45;
        wp::float32 var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        //---------
        // forward
        // def _cacc_branch(                                                                      <L 1372>
        // worldid, branchid = wp.tid()                                                           <L 1389>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 1391>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 1392>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // bodyid = body_branches[start]                                                          <L 1394>
        var_10 = wp::address(var_body_branches, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // pid = body_parentid[bodyid]                                                            <L 1395>
        var_13 = wp::address(var_body_parentid, var_11);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // local_cacc = cacc_out[worldid, pid]                                                    <L 1396>
        var_16 = wp::address(var_cacc_out, var_0, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for i in range(start, end):                                                            <L 1397>
        var_19 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_19) == 0) goto end_for_0;
            var_20 = wp::iter_next(var_19);
            // bodyid = body_branches[i]                                                          <L 1398>
            var_21 = wp::address(var_body_branches, var_20);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // dofnum = body_dofnum[bodyid]                                                       <L 1399>
            var_24 = wp::address(var_body_dofnum, var_22);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // dofadr = body_dofadr[bodyid]                                                       <L 1400>
            var_27 = wp::address(var_body_dofadr, var_22);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // for j in range(dofnum):                                                            <L 1401>
            var_30 = wp::range(var_25);
            start_for_2:;
                if (iter_cmp(var_30) == 0) goto end_for_2;
                var_31 = wp::iter_next(var_30);
                // local_cacc += cdof_dot_in[worldid, dofadr + j] * qvel_in[worldid, dofadr + j]       <L 1402>
                var_32 = wp::add(var_28, var_31);
                var_33 = wp::address(var_cdof_dot_in, var_0, var_32);
                var_34 = wp::add(var_28, var_31);
                var_35 = wp::address(var_qvel_in, var_0, var_34);
                var_37 = wp::load(var_33);
                var_38 = wp::load(var_35);
                var_36 = wp::mul(var_37, var_38);
                var_39 = wp::add(var_17, var_36);
                // if flg_acc:                                                                    <L 1403>
                if (var_flg_acc) {
                    // local_cacc += cdof_in[worldid, dofadr + j] * qacc_in[worldid, dofadr + j]       <L 1404>
                    var_40 = wp::add(var_28, var_31);
                    var_41 = wp::address(var_cdof_in, var_0, var_40);
                    var_42 = wp::add(var_28, var_31);
                    var_43 = wp::address(var_qacc_in, var_0, var_42);
                    var_45 = wp::load(var_41);
                    var_46 = wp::load(var_43);
                    var_44 = wp::mul(var_45, var_46);
                    var_47 = wp::add(var_39, var_44);
                }
                var_48 = wp::where(var_flg_acc, var_47, var_39);
                wp::assign(var_17, var_48);
                goto start_for_2;
            end_for_2:;
            // cacc_out[worldid, bodyid] = local_cacc                                             <L 1405>
            wp::array_store(var_cacc_out, var_0, var_22, var_17);
            wp::assign(var_11, var_22);
            goto start_for_0;
        end_for_0:;
    }
}

