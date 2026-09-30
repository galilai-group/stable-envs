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


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:71
static CUDA_CALLABLE void _compute_sphere_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    //---------
    // forward
    // def _compute_sphere_bounds(                                                            <L 72>
    // radius = size[0]                                                                       <L 78>
    var_1 = wp::extract(var_size, var_0);
    // return pos - wp.vec3(radius, radius, radius), pos + wp.vec3(radius, radius, radius)       <L 79>
    var_2 = wp::vec_t<3, wp::float32>(var_1, var_1, var_1);
    var_3 = wp::sub(var_pos, var_2);
    var_4 = wp::vec_t<3, wp::float32>(var_1, var_1, var_1);
    var_5 = wp::add(var_pos, var_4);
    ret_0 = var_3;
    ret_1 = var_5;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:82
static CUDA_CALLABLE void _compute_capsule_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::int32 var_10 = 2;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def _compute_capsule_bounds(                                                           <L 83>
    // radius = size[0]                                                                       <L 89>
    var_1 = wp::extract(var_size, var_0);
    // half_length = size[1]                                                                  <L 90>
    var_3 = wp::extract(var_size, var_2);
    // z = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])                                           <L 91>
    var_6 = wp::extract(var_rot, var_4, var_5);
    var_9 = wp::extract(var_rot, var_7, var_8);
    var_12 = wp::extract(var_rot, var_10, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_6, var_9, var_12);
    // world_end1 = pos - z * half_length                                                     <L 92>
    var_14 = wp::mul(var_13, var_3);
    var_15 = wp::sub(var_pos, var_14);
    // world_end2 = pos + z * half_length                                                     <L 93>
    var_16 = wp::mul(var_13, var_3);
    var_17 = wp::add(var_pos, var_16);
    // seg_min = wp.min(world_end1, world_end2)                                               <L 95>
    var_18 = wp::min(var_15, var_17);
    // seg_max = wp.max(world_end1, world_end2)                                               <L 96>
    var_19 = wp::max(var_15, var_17);
    // inflate = wp.vec3(radius, radius, radius)                                              <L 98>
    var_20 = wp::vec_t<3, wp::float32>(var_1, var_1, var_1);
    // return seg_min - inflate, seg_max + inflate                                            <L 99>
    var_21 = wp::sub(var_18, var_20);
    var_22 = wp::add(var_19, var_20);
    ret_0 = var_21;
    ret_1 = var_22;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:102
static CUDA_CALLABLE void _compute_plane_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    bool var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    const wp::float32 var_14 = 0.0;
    bool var_15;
    const wp::float32 var_16 = 1000.0;
    wp::float32 var_17;
    const wp::float32 var_18 = 10000000000.0;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = -10000000000.0;
    const wp::float32 var_21 = -10000000000.0;
    const wp::float32 var_22 = -10000000000.0;
    wp::vec_t<3, wp::float32> var_23;
    const wp::int32 var_24 = 0;
    const wp::int32 var_25 = 0;
    const wp::float32 var_26 = 2.0;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 1.0;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::float32 var_32 = 2.0;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::float32 var_35 = 1.0;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 0.0;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::int32 var_44 = 1;
    const wp::float32 var_45 = 2.0;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::float32 var_48 = 1.0;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 2.0;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 1.0;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::float32 var_57 = 0.0;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    const wp::int32 var_63 = 1;
    const wp::int32 var_64 = 0;
    const wp::float32 var_65 = 2.0;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 1.0;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 2.0;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::float32 var_74 = 1.0;
    wp::float32 var_75;
    wp::float32 var_76;
    const wp::float32 var_77 = 0.0;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    const wp::int32 var_83 = 1;
    const wp::float32 var_84 = 2.0;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::float32 var_87 = 1.0;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::float32 var_90 = 2.0;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::float32 var_93 = 1.0;
    wp::float32 var_94;
    wp::float32 var_95;
    const wp::float32 var_96 = 0.0;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    const wp::float32 var_102 = 0.01;
    const wp::float32 var_103 = 0.01;
    const wp::float32 var_104 = 0.01;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    const wp::float32 var_107 = 0.01;
    const wp::float32 var_108 = 0.01;
    const wp::float32 var_109 = 0.01;
    wp::vec_t<3, wp::float32> var_110;
    wp::vec_t<3, wp::float32> var_111;
    //---------
    // forward
    // def _compute_plane_bounds(                                                             <L 103>
    // size_scale = wp.max(size[0], size[1]) * 2.0                                            <L 110>
    var_1 = wp::extract(var_size, var_0);
    var_3 = wp::extract(var_size, var_2);
    var_4 = wp::max(var_1, var_3);
    var_6 = wp::mul(var_4, var_5);
    // if size[0] <= 0.0 or size[1] <= 0.0:                                                   <L 111>
    var_9 = wp::extract(var_size, var_8);
    var_11 = (var_9 <= var_10);
    var_7 = var_11;
    if (!var_7) {
        var_13 = wp::extract(var_size, var_12);
        var_15 = (var_13 <= var_14);
        var_7 = var_7 || var_15;
    }
    if (var_7) {
        // size_scale = 1000.0                                                                <L 112>
    }
    var_17 = wp::where(var_7, var_16, var_6);
    // min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)                                   <L 113>
    var_19 = wp::vec_t<3, wp::float32>(var_18, var_18, var_18);
    // max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)                                <L 114>
    var_23 = wp::vec_t<3, wp::float32>(var_20, var_21, var_22);
    // for i in range(2):                                                                     <L 116>
    // for j in range(2):                                                                     <L 117>
    // local_corner = wp.vec3(                                                                <L 118>
    // size_scale * (2.0 * float(i) - 1.0),                                                   <L 119>
    var_27 = wp::float(var_24);
    var_28 = wp::mul(var_26, var_27);
    var_30 = wp::sub(var_28, var_29);
    var_31 = wp::mul(var_17, var_30);
    // size_scale * (2.0 * float(j) - 1.0),                                                   <L 120>
    var_33 = wp::float(var_25);
    var_34 = wp::mul(var_32, var_33);
    var_36 = wp::sub(var_34, var_35);
    var_37 = wp::mul(var_17, var_36);
    // 0.0,                                                                                   <L 121>
    var_39 = wp::vec_t<3, wp::float32>(var_31, var_37, var_38);
    // world_corner = pos + rot @ local_corner                                                <L 123>
    var_40 = wp::mul(var_rot, var_39);
    var_41 = wp::add(var_pos, var_40);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 124>
    var_42 = wp::min(var_19, var_41);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 125>
    var_43 = wp::max(var_23, var_41);
    // local_corner = wp.vec3(                                                                <L 118>
    // size_scale * (2.0 * float(i) - 1.0),                                                   <L 119>
    var_46 = wp::float(var_24);
    var_47 = wp::mul(var_45, var_46);
    var_49 = wp::sub(var_47, var_48);
    var_50 = wp::mul(var_17, var_49);
    // size_scale * (2.0 * float(j) - 1.0),                                                   <L 120>
    var_52 = wp::float(var_44);
    var_53 = wp::mul(var_51, var_52);
    var_55 = wp::sub(var_53, var_54);
    var_56 = wp::mul(var_17, var_55);
    // 0.0,                                                                                   <L 121>
    var_58 = wp::vec_t<3, wp::float32>(var_50, var_56, var_57);
    // world_corner = pos + rot @ local_corner                                                <L 123>
    var_59 = wp::mul(var_rot, var_58);
    var_60 = wp::add(var_pos, var_59);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 124>
    var_61 = wp::min(var_42, var_60);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 125>
    var_62 = wp::max(var_43, var_60);
    // for j in range(2):                                                                     <L 117>
    // local_corner = wp.vec3(                                                                <L 118>
    // size_scale * (2.0 * float(i) - 1.0),                                                   <L 119>
    var_66 = wp::float(var_63);
    var_67 = wp::mul(var_65, var_66);
    var_69 = wp::sub(var_67, var_68);
    var_70 = wp::mul(var_17, var_69);
    // size_scale * (2.0 * float(j) - 1.0),                                                   <L 120>
    var_72 = wp::float(var_64);
    var_73 = wp::mul(var_71, var_72);
    var_75 = wp::sub(var_73, var_74);
    var_76 = wp::mul(var_17, var_75);
    // 0.0,                                                                                   <L 121>
    var_78 = wp::vec_t<3, wp::float32>(var_70, var_76, var_77);
    // world_corner = pos + rot @ local_corner                                                <L 123>
    var_79 = wp::mul(var_rot, var_78);
    var_80 = wp::add(var_pos, var_79);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 124>
    var_81 = wp::min(var_61, var_80);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 125>
    var_82 = wp::max(var_62, var_80);
    // local_corner = wp.vec3(                                                                <L 118>
    // size_scale * (2.0 * float(i) - 1.0),                                                   <L 119>
    var_85 = wp::float(var_63);
    var_86 = wp::mul(var_84, var_85);
    var_88 = wp::sub(var_86, var_87);
    var_89 = wp::mul(var_17, var_88);
    // size_scale * (2.0 * float(j) - 1.0),                                                   <L 120>
    var_91 = wp::float(var_83);
    var_92 = wp::mul(var_90, var_91);
    var_94 = wp::sub(var_92, var_93);
    var_95 = wp::mul(var_17, var_94);
    // 0.0,                                                                                   <L 121>
    var_97 = wp::vec_t<3, wp::float32>(var_89, var_95, var_96);
    // world_corner = pos + rot @ local_corner                                                <L 123>
    var_98 = wp::mul(var_rot, var_97);
    var_99 = wp::add(var_pos, var_98);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 124>
    var_100 = wp::min(var_81, var_99);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 125>
    var_101 = wp::max(var_82, var_99);
    // min_bound = min_bound - wp.vec3(0.01, 0.01, 0.01)                                      <L 127>
    var_105 = wp::vec_t<3, wp::float32>(var_102, var_103, var_104);
    var_106 = wp::sub(var_100, var_105);
    // max_bound = max_bound + wp.vec3(0.01, 0.01, 0.01)                                      <L 128>
    var_110 = wp::vec_t<3, wp::float32>(var_107, var_108, var_109);
    var_111 = wp::add(var_101, var_110);
    // return min_bound, max_bound                                                            <L 130>
    ret_0 = var_106;
    ret_1 = var_111;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:46
static CUDA_CALLABLE void _compute_box_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 10000000000.0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::float32 var_2 = -10000000000.0;
    const wp::float32 var_3 = -10000000000.0;
    const wp::float32 var_4 = -10000000000.0;
    wp::vec_t<3, wp::float32> var_5;
    const wp::int32 var_6 = 0;
    const wp::int32 var_7 = 0;
    const wp::int32 var_8 = 0;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    const wp::float32 var_11 = 2.0;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1.0;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::float32 var_19 = 2.0;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::float32 var_22 = 1.0;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    const wp::float32 var_27 = 2.0;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 1.0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    const wp::int32 var_38 = 1;
    const wp::int32 var_39 = 0;
    wp::float32 var_40;
    const wp::float32 var_41 = 2.0;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 1.0;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::float32 var_49 = 2.0;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 1.0;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    wp::float32 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 1;
    const wp::int32 var_69 = 0;
    const wp::int32 var_70 = 0;
    wp::float32 var_71;
    const wp::float32 var_72 = 2.0;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::float32 var_75 = 1.0;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::int32 var_78 = 1;
    wp::float32 var_79;
    const wp::float32 var_80 = 2.0;
    wp::float32 var_81;
    wp::float32 var_82;
    const wp::float32 var_83 = 1.0;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 2;
    wp::float32 var_87;
    const wp::float32 var_88 = 2.0;
    wp::float32 var_89;
    wp::float32 var_90;
    const wp::float32 var_91 = 1.0;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    const wp::int32 var_99 = 1;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    const wp::float32 var_102 = 2.0;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::float32 var_105 = 1.0;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::int32 var_108 = 1;
    wp::float32 var_109;
    const wp::float32 var_110 = 2.0;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::float32 var_113 = 1.0;
    wp::float32 var_114;
    wp::float32 var_115;
    const wp::int32 var_116 = 2;
    wp::float32 var_117;
    const wp::float32 var_118 = 2.0;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::float32 var_121 = 1.0;
    wp::float32 var_122;
    wp::float32 var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::vec_t<3, wp::float32> var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32> var_128;
    const wp::int32 var_129 = 1;
    const wp::int32 var_130 = 0;
    const wp::int32 var_131 = 0;
    const wp::int32 var_132 = 0;
    wp::float32 var_133;
    const wp::float32 var_134 = 2.0;
    wp::float32 var_135;
    wp::float32 var_136;
    const wp::float32 var_137 = 1.0;
    wp::float32 var_138;
    wp::float32 var_139;
    const wp::int32 var_140 = 1;
    wp::float32 var_141;
    const wp::float32 var_142 = 2.0;
    wp::float32 var_143;
    wp::float32 var_144;
    const wp::float32 var_145 = 1.0;
    wp::float32 var_146;
    wp::float32 var_147;
    const wp::int32 var_148 = 2;
    wp::float32 var_149;
    const wp::float32 var_150 = 2.0;
    wp::float32 var_151;
    wp::float32 var_152;
    const wp::float32 var_153 = 1.0;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::vec_t<3, wp::float32> var_158;
    wp::vec_t<3, wp::float32> var_159;
    wp::vec_t<3, wp::float32> var_160;
    const wp::int32 var_161 = 1;
    const wp::int32 var_162 = 0;
    wp::float32 var_163;
    const wp::float32 var_164 = 2.0;
    wp::float32 var_165;
    wp::float32 var_166;
    const wp::float32 var_167 = 1.0;
    wp::float32 var_168;
    wp::float32 var_169;
    const wp::int32 var_170 = 1;
    wp::float32 var_171;
    const wp::float32 var_172 = 2.0;
    wp::float32 var_173;
    wp::float32 var_174;
    const wp::float32 var_175 = 1.0;
    wp::float32 var_176;
    wp::float32 var_177;
    const wp::int32 var_178 = 2;
    wp::float32 var_179;
    const wp::float32 var_180 = 2.0;
    wp::float32 var_181;
    wp::float32 var_182;
    const wp::float32 var_183 = 1.0;
    wp::float32 var_184;
    wp::float32 var_185;
    wp::vec_t<3, wp::float32> var_186;
    wp::vec_t<3, wp::float32> var_187;
    wp::vec_t<3, wp::float32> var_188;
    wp::vec_t<3, wp::float32> var_189;
    wp::vec_t<3, wp::float32> var_190;
    const wp::int32 var_191 = 1;
    const wp::int32 var_192 = 0;
    const wp::int32 var_193 = 0;
    wp::float32 var_194;
    const wp::float32 var_195 = 2.0;
    wp::float32 var_196;
    wp::float32 var_197;
    const wp::float32 var_198 = 1.0;
    wp::float32 var_199;
    wp::float32 var_200;
    const wp::int32 var_201 = 1;
    wp::float32 var_202;
    const wp::float32 var_203 = 2.0;
    wp::float32 var_204;
    wp::float32 var_205;
    const wp::float32 var_206 = 1.0;
    wp::float32 var_207;
    wp::float32 var_208;
    const wp::int32 var_209 = 2;
    wp::float32 var_210;
    const wp::float32 var_211 = 2.0;
    wp::float32 var_212;
    wp::float32 var_213;
    const wp::float32 var_214 = 1.0;
    wp::float32 var_215;
    wp::float32 var_216;
    wp::vec_t<3, wp::float32> var_217;
    wp::vec_t<3, wp::float32> var_218;
    wp::vec_t<3, wp::float32> var_219;
    wp::vec_t<3, wp::float32> var_220;
    wp::vec_t<3, wp::float32> var_221;
    const wp::int32 var_222 = 1;
    const wp::int32 var_223 = 0;
    wp::float32 var_224;
    const wp::float32 var_225 = 2.0;
    wp::float32 var_226;
    wp::float32 var_227;
    const wp::float32 var_228 = 1.0;
    wp::float32 var_229;
    wp::float32 var_230;
    const wp::int32 var_231 = 1;
    wp::float32 var_232;
    const wp::float32 var_233 = 2.0;
    wp::float32 var_234;
    wp::float32 var_235;
    const wp::float32 var_236 = 1.0;
    wp::float32 var_237;
    wp::float32 var_238;
    const wp::int32 var_239 = 2;
    wp::float32 var_240;
    const wp::float32 var_241 = 2.0;
    wp::float32 var_242;
    wp::float32 var_243;
    const wp::float32 var_244 = 1.0;
    wp::float32 var_245;
    wp::float32 var_246;
    wp::vec_t<3, wp::float32> var_247;
    wp::vec_t<3, wp::float32> var_248;
    wp::vec_t<3, wp::float32> var_249;
    wp::vec_t<3, wp::float32> var_250;
    wp::vec_t<3, wp::float32> var_251;
    //---------
    // forward
    // def _compute_box_bounds(                                                               <L 47>
    // min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)                                   <L 53>
    var_1 = wp::vec_t<3, wp::float32>(var_0, var_0, var_0);
    // max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)                                <L 54>
    var_5 = wp::vec_t<3, wp::float32>(var_2, var_3, var_4);
    // for i in range(2):                                                                     <L 56>
    // for j in range(2):                                                                     <L 57>
    // for k in range(2):                                                                     <L 58>
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_10 = wp::extract(var_size, var_9);
    var_12 = wp::float(var_6);
    var_13 = wp::mul(var_11, var_12);
    var_15 = wp::sub(var_13, var_14);
    var_16 = wp::mul(var_10, var_15);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_18 = wp::extract(var_size, var_17);
    var_20 = wp::float(var_7);
    var_21 = wp::mul(var_19, var_20);
    var_23 = wp::sub(var_21, var_22);
    var_24 = wp::mul(var_18, var_23);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_26 = wp::extract(var_size, var_25);
    var_28 = wp::float(var_8);
    var_29 = wp::mul(var_27, var_28);
    var_31 = wp::sub(var_29, var_30);
    var_32 = wp::mul(var_26, var_31);
    var_33 = wp::vec_t<3, wp::float32>(var_16, var_24, var_32);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_34 = wp::mul(var_rot, var_33);
    var_35 = wp::add(var_pos, var_34);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_36 = wp::min(var_1, var_35);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_37 = wp::max(var_5, var_35);
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_40 = wp::extract(var_size, var_39);
    var_42 = wp::float(var_6);
    var_43 = wp::mul(var_41, var_42);
    var_45 = wp::sub(var_43, var_44);
    var_46 = wp::mul(var_40, var_45);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_48 = wp::extract(var_size, var_47);
    var_50 = wp::float(var_7);
    var_51 = wp::mul(var_49, var_50);
    var_53 = wp::sub(var_51, var_52);
    var_54 = wp::mul(var_48, var_53);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_56 = wp::extract(var_size, var_55);
    var_58 = wp::float(var_38);
    var_59 = wp::mul(var_57, var_58);
    var_61 = wp::sub(var_59, var_60);
    var_62 = wp::mul(var_56, var_61);
    var_63 = wp::vec_t<3, wp::float32>(var_46, var_54, var_62);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_64 = wp::mul(var_rot, var_63);
    var_65 = wp::add(var_pos, var_64);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_66 = wp::min(var_36, var_65);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_67 = wp::max(var_37, var_65);
    // for k in range(2):                                                                     <L 58>
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_71 = wp::extract(var_size, var_70);
    var_73 = wp::float(var_6);
    var_74 = wp::mul(var_72, var_73);
    var_76 = wp::sub(var_74, var_75);
    var_77 = wp::mul(var_71, var_76);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_79 = wp::extract(var_size, var_78);
    var_81 = wp::float(var_68);
    var_82 = wp::mul(var_80, var_81);
    var_84 = wp::sub(var_82, var_83);
    var_85 = wp::mul(var_79, var_84);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_87 = wp::extract(var_size, var_86);
    var_89 = wp::float(var_69);
    var_90 = wp::mul(var_88, var_89);
    var_92 = wp::sub(var_90, var_91);
    var_93 = wp::mul(var_87, var_92);
    var_94 = wp::vec_t<3, wp::float32>(var_77, var_85, var_93);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_95 = wp::mul(var_rot, var_94);
    var_96 = wp::add(var_pos, var_95);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_97 = wp::min(var_66, var_96);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_98 = wp::max(var_67, var_96);
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_101 = wp::extract(var_size, var_100);
    var_103 = wp::float(var_6);
    var_104 = wp::mul(var_102, var_103);
    var_106 = wp::sub(var_104, var_105);
    var_107 = wp::mul(var_101, var_106);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_109 = wp::extract(var_size, var_108);
    var_111 = wp::float(var_68);
    var_112 = wp::mul(var_110, var_111);
    var_114 = wp::sub(var_112, var_113);
    var_115 = wp::mul(var_109, var_114);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_117 = wp::extract(var_size, var_116);
    var_119 = wp::float(var_99);
    var_120 = wp::mul(var_118, var_119);
    var_122 = wp::sub(var_120, var_121);
    var_123 = wp::mul(var_117, var_122);
    var_124 = wp::vec_t<3, wp::float32>(var_107, var_115, var_123);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_125 = wp::mul(var_rot, var_124);
    var_126 = wp::add(var_pos, var_125);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_127 = wp::min(var_97, var_126);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_128 = wp::max(var_98, var_126);
    // for j in range(2):                                                                     <L 57>
    // for k in range(2):                                                                     <L 58>
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_133 = wp::extract(var_size, var_132);
    var_135 = wp::float(var_129);
    var_136 = wp::mul(var_134, var_135);
    var_138 = wp::sub(var_136, var_137);
    var_139 = wp::mul(var_133, var_138);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_141 = wp::extract(var_size, var_140);
    var_143 = wp::float(var_130);
    var_144 = wp::mul(var_142, var_143);
    var_146 = wp::sub(var_144, var_145);
    var_147 = wp::mul(var_141, var_146);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_149 = wp::extract(var_size, var_148);
    var_151 = wp::float(var_131);
    var_152 = wp::mul(var_150, var_151);
    var_154 = wp::sub(var_152, var_153);
    var_155 = wp::mul(var_149, var_154);
    var_156 = wp::vec_t<3, wp::float32>(var_139, var_147, var_155);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_157 = wp::mul(var_rot, var_156);
    var_158 = wp::add(var_pos, var_157);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_159 = wp::min(var_127, var_158);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_160 = wp::max(var_128, var_158);
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_163 = wp::extract(var_size, var_162);
    var_165 = wp::float(var_129);
    var_166 = wp::mul(var_164, var_165);
    var_168 = wp::sub(var_166, var_167);
    var_169 = wp::mul(var_163, var_168);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_171 = wp::extract(var_size, var_170);
    var_173 = wp::float(var_130);
    var_174 = wp::mul(var_172, var_173);
    var_176 = wp::sub(var_174, var_175);
    var_177 = wp::mul(var_171, var_176);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_179 = wp::extract(var_size, var_178);
    var_181 = wp::float(var_161);
    var_182 = wp::mul(var_180, var_181);
    var_184 = wp::sub(var_182, var_183);
    var_185 = wp::mul(var_179, var_184);
    var_186 = wp::vec_t<3, wp::float32>(var_169, var_177, var_185);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_187 = wp::mul(var_rot, var_186);
    var_188 = wp::add(var_pos, var_187);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_189 = wp::min(var_159, var_188);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_190 = wp::max(var_160, var_188);
    // for k in range(2):                                                                     <L 58>
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_194 = wp::extract(var_size, var_193);
    var_196 = wp::float(var_129);
    var_197 = wp::mul(var_195, var_196);
    var_199 = wp::sub(var_197, var_198);
    var_200 = wp::mul(var_194, var_199);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_202 = wp::extract(var_size, var_201);
    var_204 = wp::float(var_191);
    var_205 = wp::mul(var_203, var_204);
    var_207 = wp::sub(var_205, var_206);
    var_208 = wp::mul(var_202, var_207);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_210 = wp::extract(var_size, var_209);
    var_212 = wp::float(var_192);
    var_213 = wp::mul(var_211, var_212);
    var_215 = wp::sub(var_213, var_214);
    var_216 = wp::mul(var_210, var_215);
    var_217 = wp::vec_t<3, wp::float32>(var_200, var_208, var_216);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_218 = wp::mul(var_rot, var_217);
    var_219 = wp::add(var_pos, var_218);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_220 = wp::min(var_189, var_219);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_221 = wp::max(var_190, var_219);
    // local_corner = wp.vec3(                                                                <L 59>
    // size[0] * (2.0 * float(i) - 1.0),                                                      <L 60>
    var_224 = wp::extract(var_size, var_223);
    var_226 = wp::float(var_129);
    var_227 = wp::mul(var_225, var_226);
    var_229 = wp::sub(var_227, var_228);
    var_230 = wp::mul(var_224, var_229);
    // size[1] * (2.0 * float(j) - 1.0),                                                      <L 61>
    var_232 = wp::extract(var_size, var_231);
    var_234 = wp::float(var_191);
    var_235 = wp::mul(var_233, var_234);
    var_237 = wp::sub(var_235, var_236);
    var_238 = wp::mul(var_232, var_237);
    // size[2] * (2.0 * float(k) - 1.0),                                                      <L 62>
    var_240 = wp::extract(var_size, var_239);
    var_242 = wp::float(var_222);
    var_243 = wp::mul(var_241, var_242);
    var_245 = wp::sub(var_243, var_244);
    var_246 = wp::mul(var_240, var_245);
    var_247 = wp::vec_t<3, wp::float32>(var_230, var_238, var_246);
    // world_corner = pos + rot @ local_corner                                                <L 64>
    var_248 = wp::mul(var_rot, var_247);
    var_249 = wp::add(var_pos, var_248);
    // min_bound = wp.min(min_bound, world_corner)                                            <L 65>
    var_250 = wp::min(var_220, var_249);
    // max_bound = wp.max(max_bound, world_corner)                                            <L 66>
    var_251 = wp::max(var_221, var_249);
    // return min_bound, max_bound                                                            <L 68>
    ret_0 = var_250;
    ret_1 = var_251;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:133
static CUDA_CALLABLE void _compute_ellipsoid_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    const wp::int32 var_15 = 2;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 1;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    const wp::int32 var_32 = 2;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    const wp::int32 var_38 = 2;
    const wp::int32 var_39 = 0;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 2;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::int32 var_50 = 2;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    const wp::int32 var_53 = 2;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    //---------
    // forward
    // def _compute_ellipsoid_bounds(                                                         <L 134>
    // row0 = wp.vec3(rot[0, 0] * size[0], rot[0, 1] * size[1], rot[0, 2] * size[2])          <L 141>
    var_2 = wp::extract(var_rot, var_0, var_1);
    var_4 = wp::extract(var_size, var_3);
    var_5 = wp::mul(var_2, var_4);
    var_8 = wp::extract(var_rot, var_6, var_7);
    var_10 = wp::extract(var_size, var_9);
    var_11 = wp::mul(var_8, var_10);
    var_14 = wp::extract(var_rot, var_12, var_13);
    var_16 = wp::extract(var_size, var_15);
    var_17 = wp::mul(var_14, var_16);
    var_18 = wp::vec_t<3, wp::float32>(var_5, var_11, var_17);
    // row1 = wp.vec3(rot[1, 0] * size[0], rot[1, 1] * size[1], rot[1, 2] * size[2])          <L 142>
    var_21 = wp::extract(var_rot, var_19, var_20);
    var_23 = wp::extract(var_size, var_22);
    var_24 = wp::mul(var_21, var_23);
    var_27 = wp::extract(var_rot, var_25, var_26);
    var_29 = wp::extract(var_size, var_28);
    var_30 = wp::mul(var_27, var_29);
    var_33 = wp::extract(var_rot, var_31, var_32);
    var_35 = wp::extract(var_size, var_34);
    var_36 = wp::mul(var_33, var_35);
    var_37 = wp::vec_t<3, wp::float32>(var_24, var_30, var_36);
    // row2 = wp.vec3(rot[2, 0] * size[0], rot[2, 1] * size[1], rot[2, 2] * size[2])          <L 143>
    var_40 = wp::extract(var_rot, var_38, var_39);
    var_42 = wp::extract(var_size, var_41);
    var_43 = wp::mul(var_40, var_42);
    var_46 = wp::extract(var_rot, var_44, var_45);
    var_48 = wp::extract(var_size, var_47);
    var_49 = wp::mul(var_46, var_48);
    var_52 = wp::extract(var_rot, var_50, var_51);
    var_54 = wp::extract(var_size, var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::vec_t<3, wp::float32>(var_43, var_49, var_55);
    // extent = wp.vec3(wp.length(row0), wp.length(row1), wp.length(row2))                    <L 144>
    var_57 = wp::length(var_18);
    var_58 = wp::length(var_37);
    var_59 = wp::length(var_56);
    var_60 = wp::vec_t<3, wp::float32>(var_57, var_58, var_59);
    // return pos - extent, pos + extent                                                      <L 145>
    var_61 = wp::sub(var_pos, var_60);
    var_62 = wp::add(var_pos, var_60);
    ret_0 = var_61;
    ret_1 = var_62;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:148
static CUDA_CALLABLE void _compute_cylinder_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::int32 var_10 = 2;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 2;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::int32 var_24 = 0;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    const wp::int32 var_30 = 2;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    const wp::int32 var_35 = 1;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    const wp::int32 var_41 = 1;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 1;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    const wp::int32 var_70 = 2;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 2;
    wp::float32 var_76;
    const wp::int32 var_77 = 2;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    const wp::int32 var_83 = 0;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 1;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    const wp::int32 var_91 = 2;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    //---------
    // forward
    // def _compute_cylinder_bounds(                                                          <L 149>
    // radius = size[0]                                                                       <L 155>
    var_1 = wp::extract(var_size, var_0);
    // half_height = size[1]                                                                  <L 156>
    var_3 = wp::extract(var_size, var_2);
    // axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])                                        <L 158>
    var_6 = wp::extract(var_rot, var_4, var_5);
    var_9 = wp::extract(var_rot, var_7, var_8);
    var_12 = wp::extract(var_rot, var_10, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_6, var_9, var_12);
    // axis_abs = wp.vec3(wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2]))                  <L 159>
    var_15 = wp::extract(var_13, var_14);
    var_16 = wp::abs(var_15);
    var_18 = wp::extract(var_13, var_17);
    var_19 = wp::abs(var_18);
    var_21 = wp::extract(var_13, var_20);
    var_22 = wp::abs(var_21);
    var_23 = wp::vec_t<3, wp::float32>(var_16, var_19, var_22);
    // basis_x = wp.vec3(rot[0, 0], rot[1, 0], rot[2, 0])                                     <L 161>
    var_26 = wp::extract(var_rot, var_24, var_25);
    var_29 = wp::extract(var_rot, var_27, var_28);
    var_32 = wp::extract(var_rot, var_30, var_31);
    var_33 = wp::vec_t<3, wp::float32>(var_26, var_29, var_32);
    // basis_y = wp.vec3(rot[0, 1], rot[1, 1], rot[2, 1])                                     <L 162>
    var_36 = wp::extract(var_rot, var_34, var_35);
    var_39 = wp::extract(var_rot, var_37, var_38);
    var_42 = wp::extract(var_rot, var_40, var_41);
    var_43 = wp::vec_t<3, wp::float32>(var_36, var_39, var_42);
    // radial_x = radius * wp.sqrt(basis_x[0] * basis_x[0] + basis_y[0] * basis_y[0])         <L 164>
    var_45 = wp::extract(var_33, var_44);
    var_47 = wp::extract(var_33, var_46);
    var_48 = wp::mul(var_45, var_47);
    var_50 = wp::extract(var_43, var_49);
    var_52 = wp::extract(var_43, var_51);
    var_53 = wp::mul(var_50, var_52);
    var_54 = wp::add(var_48, var_53);
    var_55 = wp::sqrt(var_54);
    var_56 = wp::mul(var_1, var_55);
    // radial_y = radius * wp.sqrt(basis_x[1] * basis_x[1] + basis_y[1] * basis_y[1])         <L 165>
    var_58 = wp::extract(var_33, var_57);
    var_60 = wp::extract(var_33, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_63 = wp::extract(var_43, var_62);
    var_65 = wp::extract(var_43, var_64);
    var_66 = wp::mul(var_63, var_65);
    var_67 = wp::add(var_61, var_66);
    var_68 = wp::sqrt(var_67);
    var_69 = wp::mul(var_1, var_68);
    // radial_z = radius * wp.sqrt(basis_x[2] * basis_x[2] + basis_y[2] * basis_y[2])         <L 166>
    var_71 = wp::extract(var_33, var_70);
    var_73 = wp::extract(var_33, var_72);
    var_74 = wp::mul(var_71, var_73);
    var_76 = wp::extract(var_43, var_75);
    var_78 = wp::extract(var_43, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::add(var_74, var_79);
    var_81 = wp::sqrt(var_80);
    var_82 = wp::mul(var_1, var_81);
    // extent = wp.vec3(                                                                      <L 168>
    // radial_x + half_height * axis_abs[0],                                                  <L 169>
    var_84 = wp::extract(var_23, var_83);
    var_85 = wp::mul(var_3, var_84);
    var_86 = wp::add(var_56, var_85);
    // radial_y + half_height * axis_abs[1],                                                  <L 170>
    var_88 = wp::extract(var_23, var_87);
    var_89 = wp::mul(var_3, var_88);
    var_90 = wp::add(var_69, var_89);
    // radial_z + half_height * axis_abs[2],                                                  <L 171>
    var_92 = wp::extract(var_23, var_91);
    var_93 = wp::mul(var_3, var_92);
    var_94 = wp::add(var_82, var_93);
    var_95 = wp::vec_t<3, wp::float32>(var_86, var_90, var_94);
    // return pos - extent, pos + extent                                                      <L 174>
    var_96 = wp::sub(var_pos, var_95);
    var_97 = wp::add(var_pos, var_95);
    ret_0 = var_96;
    ret_1 = var_97;
    return;
}


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


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:71
static CUDA_CALLABLE void adj__compute_sphere_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:82
static CUDA_CALLABLE void adj__compute_capsule_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:102
static CUDA_CALLABLE void adj__compute_plane_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:46
static CUDA_CALLABLE void adj__compute_box_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:133
static CUDA_CALLABLE void adj__compute_ellipsoid_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/bvh.py:148
static CUDA_CALLABLE void adj__compute_cylinder_bounds_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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



extern "C" __global__ void compute_mesh_group_roots_0cd108f4_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::uint64 var_mesh_id,
    wp::array_t<wp::int32> var_group_root_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        //---------
        // forward
        // def compute_mesh_group_roots(                                                          <L 254>
        // tid = wp.tid()                                                                         <L 260>
        var_0 = builtin_tid1d();
        // root = wp.mesh_get_group_root(mesh_id, tid)                                            <L 261>
        var_1 = wp::mesh_get_group_root(var_mesh_id, var_0);
        // group_root_out[tid] = root                                                             <L 262>
        wp::array_store(var_group_root_out, var_0, var_1);
}



extern "C" __global__ void _compute_flex_bvh_bounds_1b1d0760_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_vertnum,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_total_bvh_size,
    wp::array_t<wp::vec_t<3, wp::float32>> var_lower_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_upper_out,
    wp::array_t<wp::int32> var_group_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
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
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::vec_t<3, wp::float32> var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::float32 var_45 = 10000000000.0;
        wp::vec_t<3, wp::float32> var_46;
        const wp::float32 var_47 = -10000000000.0;
        const wp::float32 var_48 = -10000000000.0;
        const wp::float32 var_49 = -10000000000.0;
        wp::vec_t<3, wp::float32> var_50;
        wp::range_t var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::int32 var_61;
        //---------
        // forward
        // def _compute_flex_bvh_bounds(                                                          <L 266>
        // worldid, flexlocalid = wp.tid()                                                        <L 284>
        builtin_tid2d(var_0, var_1);
        // flex_id = flex_geom_flexid[flexlocalid]                                                <L 286>
        var_2 = wp::address(var_flex_geom_flexid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // edge_id = flex_geom_edgeid[flexlocalid]                                                <L 287>
        var_5 = wp::address(var_flex_geom_edgeid, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // out_idx = worldid * total_bvh_size + bvh_ngeom + flexlocalid                           <L 288>
        var_8 = wp::mul(var_0, var_total_bvh_size);
        var_9 = wp::add(var_8, var_bvh_ngeom);
        var_10 = wp::add(var_9, var_1);
        // radius = flex_radius[flex_id]                                                          <L 289>
        var_11 = wp::address(var_flex_radius, var_3);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // inflate = wp.vec3(radius, radius, radius)                                              <L 290>
        var_14 = wp::vec_t<3, wp::float32>(var_12, var_12, var_12);
        // if edge_id >= 0:  # capsule (1D edge)                                                  <L 292>
        var_16 = (var_6 >= var_15);
        if (var_16) {
            // edge = flex_edge[edge_id]                                                          <L 293>
            var_17 = wp::address(var_flex_edge, var_6);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // vert_adr = flex_vertadr[flex_id]                                                   <L 294>
            var_20 = wp::address(var_flex_vertadr, var_3);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // v0 = flexvert_xpos_in[worldid, vert_adr + edge[0]]                                 <L 295>
            var_24 = wp::extract(var_18, var_23);
            var_25 = wp::add(var_21, var_24);
            var_26 = wp::address(var_flexvert_xpos_in, var_0, var_25);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // v1 = flexvert_xpos_in[worldid, vert_adr + edge[1]]                                 <L 296>
            var_30 = wp::extract(var_18, var_29);
            var_31 = wp::add(var_21, var_30);
            var_32 = wp::address(var_flexvert_xpos_in, var_0, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // lower_out[out_idx] = wp.min(v0, v1) - inflate                                      <L 297>
            var_35 = wp::min(var_27, var_33);
            var_36 = wp::sub(var_35, var_14);
            wp::array_store(var_lower_out, var_10, var_36);
            // upper_out[out_idx] = wp.max(v0, v1) + inflate                                      <L 298>
            var_37 = wp::max(var_27, var_33);
            var_38 = wp::add(var_37, var_14);
            wp::array_store(var_upper_out, var_10, var_38);
        }
        if (!var_16) {
            // vert_adr = flex_vertadr[flex_id]                                                   <L 300>
            var_39 = wp::address(var_flex_vertadr, var_3);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // nvert = flex_vertnum[flex_id]                                                      <L 301>
            var_42 = wp::address(var_flex_vertnum, var_3);
            var_44 = wp::load(var_42);
            var_43 = wp::copy(var_44);
            // min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)                               <L 302>
            var_46 = wp::vec_t<3, wp::float32>(var_45, var_45, var_45);
            // max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)                            <L 303>
            var_50 = wp::vec_t<3, wp::float32>(var_47, var_48, var_49);
            // for i in range(nvert):                                                             <L 304>
            var_51 = wp::range(var_43);
            start_for_0:;
                if (iter_cmp(var_51) == 0) goto end_for_0;
                var_52 = wp::iter_next(var_51);
                // v = flexvert_xpos_in[worldid, vert_adr + i]                                    <L 305>
                var_53 = wp::add(var_40, var_52);
                var_54 = wp::address(var_flexvert_xpos_in, var_0, var_53);
                var_56 = wp::load(var_54);
                var_55 = wp::copy(var_56);
                // min_bound = wp.min(min_bound, v)                                               <L 306>
                var_57 = wp::min(var_46, var_55);
                // max_bound = wp.max(max_bound, v)                                               <L 307>
                var_58 = wp::max(var_50, var_55);
                wp::assign(var_46, var_57);
                wp::assign(var_50, var_58);
                goto start_for_0;
            end_for_0:;
            // lower_out[out_idx] = min_bound - inflate                                           <L 308>
            var_59 = wp::sub(var_46, var_14);
            wp::array_store(var_lower_out, var_10, var_59);
            // upper_out[out_idx] = max_bound + inflate                                           <L 309>
            var_60 = wp::add(var_50, var_14);
            wp::array_store(var_upper_out, var_10, var_60);
        }
        var_61 = wp::where(var_16, var_21, var_40);
        // group_out[out_idx] = worldid                                                           <L 311>
        wp::array_store(var_group_out, var_10, var_0);
}



extern "C" __global__ void _update_flex_2d_face_points_1db70403_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_elemnum,
    wp::array_t<wp::int32> var_flex_elemdataadr,
    wp::array_t<wp::int32> var_flex_shelldataadr,
    wp::array_t<wp::int32> var_flex_elem,
    wp::array_t<wp::int32> var_flex_shell,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_norm_in,
    wp::int32 var_flex_id,
    wp::int32 var_nface,
    bool var_smooth,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_point_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
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
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        bool var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 3;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 2;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32>* var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        const wp::int32 var_75 = 2;
        wp::int32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 3;
        wp::int32 var_79;
        const wp::int32 var_80 = 0;
        wp::int32 var_81;
        const wp::int32 var_82 = 1;
        wp::int32 var_83;
        const wp::int32 var_84 = 2;
        wp::int32 var_85;
        const wp::int32 var_86 = 2;
        wp::int32 var_87;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        wp::int32 var_90;
        const wp::int32 var_91 = 3;
        wp::int32 var_92;
        const wp::int32 var_93 = 0;
        wp::int32 var_94;
        const wp::int32 var_95 = 1;
        wp::int32 var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        const wp::int32 var_103 = 2;
        wp::int32 var_104;
        wp::int32 var_105;
        const wp::int32 var_106 = 0;
        wp::int32 var_107;
        wp::int32* var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        const wp::int32 var_111 = 1;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32>* var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32>* var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32>* var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::vec_t<3, wp::float32>* var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        const wp::int32 var_128 = 2;
        wp::int32 var_129;
        wp::int32 var_130;
        const wp::int32 var_131 = 2;
        wp::int32 var_132;
        wp::int32 var_133;
        const wp::int32 var_134 = 3;
        wp::int32 var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        const wp::int32 var_138 = 0;
        wp::int32 var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        const wp::int32 var_142 = 1;
        wp::int32 var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::vec_t<3, wp::float32> var_145;
        const wp::int32 var_146 = 2;
        wp::int32 var_147;
        wp::int32 var_148;
        const wp::int32 var_149 = 2;
        wp::int32 var_150;
        const wp::int32 var_151 = 1;
        wp::int32 var_152;
        wp::int32 var_153;
        const wp::int32 var_154 = 3;
        wp::int32 var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        const wp::int32 var_158 = 0;
        wp::int32 var_159;
        wp::vec_t<3, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        const wp::int32 var_162 = 1;
        wp::int32 var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        const wp::int32 var_166 = 2;
        wp::int32 var_167;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::vec_t<3, wp::float32> var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        //---------
        // forward
        // def _update_flex_2d_face_points(                                                       <L 864>
        // worldid, workid = wp.tid()                                                             <L 883>
        builtin_tid2d(var_0, var_1);
        // elem_adr = flex_elemdataadr[flex_id]                                                   <L 885>
        var_2 = wp::address(var_flex_elemdataadr, var_flex_id);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // vert_adr = flex_vertadr[flex_id]                                                       <L 886>
        var_5 = wp::address(var_flex_vertadr, var_flex_id);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // radius = flex_radius[flex_id]                                                          <L 887>
        var_8 = wp::address(var_flex_radius, var_flex_id);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // nelem = flex_elemnum[flex_id]                                                          <L 888>
        var_11 = wp::address(var_flex_elemnum, var_flex_id);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // world_face_offset = worldid * nface                                                    <L 889>
        var_14 = wp::mul(var_0, var_nface);
        // if workid < nelem:                                                                     <L 891>
        var_15 = (var_1 < var_12);
        if (var_15) {
            // elemid = workid                                                                    <L 893>
            var_16 = wp::copy(var_1);
            // ebase = elem_adr + elemid * 3                                                      <L 894>
            var_18 = wp::mul(var_16, var_17);
            var_19 = wp::add(var_3, var_18);
            // i0 = vert_adr + flex_elem[ebase + 0]                                               <L 895>
            var_21 = wp::add(var_19, var_20);
            var_22 = wp::address(var_flex_elem, var_21);
            var_24 = wp::load(var_22);
            var_23 = wp::add(var_6, var_24);
            // i1 = vert_adr + flex_elem[ebase + 1]                                               <L 896>
            var_26 = wp::add(var_19, var_25);
            var_27 = wp::address(var_flex_elem, var_26);
            var_29 = wp::load(var_27);
            var_28 = wp::add(var_6, var_29);
            // i2 = vert_adr + flex_elem[ebase + 2]                                               <L 897>
            var_31 = wp::add(var_19, var_30);
            var_32 = wp::address(var_flex_elem, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::add(var_6, var_34);
            // v0 = flexvert_xpos_in[worldid, i0]                                                 <L 899>
            var_35 = wp::address(var_flexvert_xpos_in, var_0, var_23);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // v1 = flexvert_xpos_in[worldid, i1]                                                 <L 900>
            var_38 = wp::address(var_flexvert_xpos_in, var_0, var_28);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // v2 = flexvert_xpos_in[worldid, i2]                                                 <L 901>
            var_41 = wp::address(var_flexvert_xpos_in, var_0, var_33);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // if smooth:                                                                         <L 904>
            if (var_smooth) {
                // n0 = flexvert_norm_in[worldid, i0]                                             <L 905>
                var_44 = wp::address(var_flexvert_norm_in, var_0, var_23);
                var_46 = wp::load(var_44);
                var_45 = wp::copy(var_46);
                // n1 = flexvert_norm_in[worldid, i1]                                             <L 906>
                var_47 = wp::address(var_flexvert_norm_in, var_0, var_28);
                var_49 = wp::load(var_47);
                var_48 = wp::copy(var_49);
                // n2 = flexvert_norm_in[worldid, i2]                                             <L 907>
                var_50 = wp::address(var_flexvert_norm_in, var_0, var_33);
                var_52 = wp::load(var_50);
                var_51 = wp::copy(var_52);
            }
            if (!var_smooth) {
                // face_nrm = wp.cross(v1 - v0, v2 - v0)                                          <L 909>
                var_53 = wp::sub(var_39, var_36);
                var_54 = wp::sub(var_42, var_36);
                var_55 = wp::cross(var_53, var_54);
                // face_nrm = wp.normalize(face_nrm)                                              <L 910>
                var_56 = wp::normalize(var_55);
                // n0 = face_nrm                                                                  <L 911>
                var_57 = wp::copy(var_56);
                // n1 = face_nrm                                                                  <L 912>
                var_58 = wp::copy(var_56);
                // n2 = face_nrm                                                                  <L 913>
                var_59 = wp::copy(var_56);
            }
            var_60 = wp::where(var_smooth, var_45, var_57);
            var_61 = wp::where(var_smooth, var_48, var_58);
            var_62 = wp::where(var_smooth, var_51, var_59);
            // p0_pos = v0 + radius * n0                                                          <L 915>
            var_63 = wp::mul(var_9, var_60);
            var_64 = wp::add(var_36, var_63);
            // p1_pos = v1 + radius * n1                                                          <L 916>
            var_65 = wp::mul(var_9, var_61);
            var_66 = wp::add(var_39, var_65);
            // p2_pos = v2 + radius * n2                                                          <L 917>
            var_67 = wp::mul(var_9, var_62);
            var_68 = wp::add(var_42, var_67);
            // p0_neg = v0 - radius * n0                                                          <L 919>
            var_69 = wp::mul(var_9, var_60);
            var_70 = wp::sub(var_36, var_69);
            // p1_neg = v1 - radius * n1                                                          <L 920>
            var_71 = wp::mul(var_9, var_61);
            var_72 = wp::sub(var_39, var_71);
            // p2_neg = v2 - radius * n2                                                          <L 921>
            var_73 = wp::mul(var_9, var_62);
            var_74 = wp::sub(var_42, var_73);
            // face_id0 = world_face_offset + (2 * elemid)                                        <L 923>
            var_76 = wp::mul(var_75, var_16);
            var_77 = wp::add(var_14, var_76);
            // base0 = face_id0 * 3                                                               <L 924>
            var_79 = wp::mul(var_77, var_78);
            // face_point_out[base0 + 0] = p0_pos                                                 <L 925>
            var_81 = wp::add(var_79, var_80);
            wp::array_store(var_face_point_out, var_81, var_64);
            // face_point_out[base0 + 1] = p1_pos                                                 <L 926>
            var_83 = wp::add(var_79, var_82);
            wp::array_store(var_face_point_out, var_83, var_66);
            // face_point_out[base0 + 2] = p2_pos                                                 <L 927>
            var_85 = wp::add(var_79, var_84);
            wp::array_store(var_face_point_out, var_85, var_68);
            // face_id1 = world_face_offset + (2 * elemid + 1)                                    <L 929>
            var_87 = wp::mul(var_86, var_16);
            var_89 = wp::add(var_87, var_88);
            var_90 = wp::add(var_14, var_89);
            // base1 = face_id1 * 3                                                               <L 930>
            var_92 = wp::mul(var_90, var_91);
            // face_point_out[base1 + 0] = p0_neg                                                 <L 931>
            var_94 = wp::add(var_92, var_93);
            wp::array_store(var_face_point_out, var_94, var_70);
            // face_point_out[base1 + 1] = p1_neg                                                 <L 932>
            var_96 = wp::add(var_92, var_95);
            wp::array_store(var_face_point_out, var_96, var_72);
            // face_point_out[base1 + 2] = p2_neg                                                 <L 933>
            var_98 = wp::add(var_92, var_97);
            wp::array_store(var_face_point_out, var_98, var_74);
        }
        if (!var_15) {
            // shell_adr = flex_shelldataadr[flex_id]                                             <L 936>
            var_99 = wp::address(var_flex_shelldataadr, var_flex_id);
            var_101 = wp::load(var_99);
            var_100 = wp::copy(var_101);
            // shellid = workid - nelem                                                           <L 937>
            var_102 = wp::sub(var_1, var_12);
            // sbase = shell_adr + 2 * shellid                                                    <L 938>
            var_104 = wp::mul(var_103, var_102);
            var_105 = wp::add(var_100, var_104);
            // i0 = vert_adr + flex_shell[sbase + 0]                                              <L 939>
            var_107 = wp::add(var_105, var_106);
            var_108 = wp::address(var_flex_shell, var_107);
            var_110 = wp::load(var_108);
            var_109 = wp::add(var_6, var_110);
            // i1 = vert_adr + flex_shell[sbase + 1]                                              <L 940>
            var_112 = wp::add(var_105, var_111);
            var_113 = wp::address(var_flex_shell, var_112);
            var_115 = wp::load(var_113);
            var_114 = wp::add(var_6, var_115);
            // v0 = flexvert_xpos_in[worldid, i0]                                                 <L 942>
            var_116 = wp::address(var_flexvert_xpos_in, var_0, var_109);
            var_118 = wp::load(var_116);
            var_117 = wp::copy(var_118);
            // v1 = flexvert_xpos_in[worldid, i1]                                                 <L 943>
            var_119 = wp::address(var_flexvert_xpos_in, var_0, var_114);
            var_121 = wp::load(var_119);
            var_120 = wp::copy(var_121);
            // n0 = flexvert_norm_in[worldid, i0]                                                 <L 945>
            var_122 = wp::address(var_flexvert_norm_in, var_0, var_109);
            var_124 = wp::load(var_122);
            var_123 = wp::copy(var_124);
            // n1 = flexvert_norm_in[worldid, i1]                                                 <L 946>
            var_125 = wp::address(var_flexvert_norm_in, var_0, var_114);
            var_127 = wp::load(var_125);
            var_126 = wp::copy(var_127);
            // shell_face_offset = 2 * nelem                                                      <L 948>
            var_129 = wp::mul(var_128, var_12);
            // face_id0 = world_face_offset + shell_face_offset + (2 * shellid)                   <L 949>
            var_130 = wp::add(var_14, var_129);
            var_132 = wp::mul(var_131, var_102);
            var_133 = wp::add(var_130, var_132);
            // base0 = face_id0 * 3                                                               <L 950>
            var_135 = wp::mul(var_133, var_134);
            // face_point_out[base0 + 0] = v0 + radius * n0                                       <L 951>
            var_136 = wp::mul(var_9, var_123);
            var_137 = wp::add(var_117, var_136);
            var_139 = wp::add(var_135, var_138);
            wp::array_store(var_face_point_out, var_139, var_137);
            // face_point_out[base0 + 1] = v1 - radius * n1                                       <L 952>
            var_140 = wp::mul(var_9, var_126);
            var_141 = wp::sub(var_120, var_140);
            var_143 = wp::add(var_135, var_142);
            wp::array_store(var_face_point_out, var_143, var_141);
            // face_point_out[base0 + 2] = v1 + radius * n1                                       <L 953>
            var_144 = wp::mul(var_9, var_126);
            var_145 = wp::add(var_120, var_144);
            var_147 = wp::add(var_135, var_146);
            wp::array_store(var_face_point_out, var_147, var_145);
            // face_id1 = world_face_offset + shell_face_offset + (2 * shellid + 1)               <L 955>
            var_148 = wp::add(var_14, var_129);
            var_150 = wp::mul(var_149, var_102);
            var_152 = wp::add(var_150, var_151);
            var_153 = wp::add(var_148, var_152);
            // base1 = face_id1 * 3                                                               <L 956>
            var_155 = wp::mul(var_153, var_154);
            // face_point_out[base1 + 0] = v1 - radius * n1                                       <L 957>
            var_156 = wp::mul(var_9, var_126);
            var_157 = wp::sub(var_120, var_156);
            var_159 = wp::add(var_155, var_158);
            wp::array_store(var_face_point_out, var_159, var_157);
            // face_point_out[base1 + 1] = v0 + radius * n0                                       <L 958>
            var_160 = wp::mul(var_9, var_123);
            var_161 = wp::add(var_117, var_160);
            var_163 = wp::add(var_155, var_162);
            wp::array_store(var_face_point_out, var_163, var_161);
            // face_point_out[base1 + 2] = v0 - radius * n0                                       <L 959>
            var_164 = wp::mul(var_9, var_123);
            var_165 = wp::sub(var_117, var_164);
            var_167 = wp::add(var_155, var_166);
            wp::array_store(var_face_point_out, var_167, var_165);
        }
        var_168 = wp::where(var_15, var_23, var_109);
        var_169 = wp::where(var_15, var_28, var_114);
        var_170 = wp::where(var_15, var_36, var_117);
        var_171 = wp::where(var_15, var_39, var_120);
        var_172 = wp::where(var_15, var_60, var_123);
        var_173 = wp::where(var_15, var_61, var_126);
        var_174 = wp::where(var_15, var_77, var_133);
        var_175 = wp::where(var_15, var_79, var_135);
        var_176 = wp::where(var_15, var_90, var_153);
        var_177 = wp::where(var_15, var_92, var_155);
}



extern "C" __global__ void normalize_vertex_normals_304dd665_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_norm_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        //---------
        // forward
        // def normalize_vertex_normals(                                                          <L 672>
        // worldid, vertid = wp.tid()                                                             <L 677>
        builtin_tid2d(var_0, var_1);
        // flexvert_norm_out[worldid, vertid] = wp.normalize(flexvert_norm_out[worldid, vertid])       <L 678>
        var_2 = wp::address(var_flexvert_norm_out, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::normalize(var_4);
        wp::array_store(var_flexvert_norm_out, var_0, var_1, var_3);
}



extern "C" __global__ void _build_flex_3d_shells_4ecf1171_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_shell,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::int32 var_shell_adr,
    wp::int32 var_vert_adr,
    wp::int32 var_face_offset,
    wp::int32 var_nface,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_point_out,
    wp::array_t<wp::int32> var_face_index_out,
    wp::array_t<wp::int32> var_group_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 3;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 3;
        wp::int32 var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        const wp::int32 var_48 = 2;
        wp::int32 var_49;
        const wp::int32 var_50 = 2;
        wp::int32 var_51;
        //---------
        // forward
        // def _build_flex_3d_shells(                                                             <L 818>
        // worldid, shellid = wp.tid()                                                            <L 838>
        builtin_tid2d(var_0, var_1);
        // base = shell_adr + shellid * 3                                                         <L 840>
        var_3 = wp::mul(var_1, var_2);
        var_4 = wp::add(var_shell_adr, var_3);
        // i0 = vert_adr + flex_shell[base + 0]                                                   <L 841>
        var_6 = wp::add(var_4, var_5);
        var_7 = wp::address(var_flex_shell, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::add(var_vert_adr, var_9);
        // i1 = vert_adr + flex_shell[base + 1]                                                   <L 842>
        var_11 = wp::add(var_4, var_10);
        var_12 = wp::address(var_flex_shell, var_11);
        var_14 = wp::load(var_12);
        var_13 = wp::add(var_vert_adr, var_14);
        // i2 = vert_adr + flex_shell[base + 2]                                                   <L 843>
        var_16 = wp::add(var_4, var_15);
        var_17 = wp::address(var_flex_shell, var_16);
        var_19 = wp::load(var_17);
        var_18 = wp::add(var_vert_adr, var_19);
        // face_id = worldid * nface + face_offset + shellid                                      <L 845>
        var_20 = wp::mul(var_0, var_nface);
        var_21 = wp::add(var_20, var_face_offset);
        var_22 = wp::add(var_21, var_1);
        // base = face_id * 3                                                                     <L 846>
        var_24 = wp::mul(var_22, var_23);
        // v0 = flexvert_xpos_in[worldid, i0]                                                     <L 848>
        var_25 = wp::address(var_flexvert_xpos_in, var_0, var_8);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // v1 = flexvert_xpos_in[worldid, i1]                                                     <L 849>
        var_28 = wp::address(var_flexvert_xpos_in, var_0, var_13);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // v2 = flexvert_xpos_in[worldid, i2]                                                     <L 850>
        var_31 = wp::address(var_flexvert_xpos_in, var_0, var_18);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // face_point_out[base + 0] = v0                                                          <L 852>
        var_35 = wp::add(var_24, var_34);
        wp::array_store(var_face_point_out, var_35, var_26);
        // face_point_out[base + 1] = v1                                                          <L 853>
        var_37 = wp::add(var_24, var_36);
        wp::array_store(var_face_point_out, var_37, var_29);
        // face_point_out[base + 2] = v2                                                          <L 854>
        var_39 = wp::add(var_24, var_38);
        wp::array_store(var_face_point_out, var_39, var_32);
        // face_index_out[base + 0] = base + 0                                                    <L 856>
        var_41 = wp::add(var_24, var_40);
        var_43 = wp::add(var_24, var_42);
        wp::array_store(var_face_index_out, var_43, var_41);
        // face_index_out[base + 1] = base + 1                                                    <L 857>
        var_45 = wp::add(var_24, var_44);
        var_47 = wp::add(var_24, var_46);
        wp::array_store(var_face_index_out, var_47, var_45);
        // face_index_out[base + 2] = base + 2                                                    <L 858>
        var_49 = wp::add(var_24, var_48);
        var_51 = wp::add(var_24, var_50);
        wp::array_store(var_face_index_out, var_51, var_49);
        // group_out[face_id] = worldid                                                           <L 860>
        wp::array_store(var_group_out, var_22, var_0);
}



extern "C" __global__ void _build_flex_2d_sides_5b45004a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_shell,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_norm_in,
    wp::int32 var_shell_adr,
    wp::int32 var_vert_adr,
    wp::int32 var_face_offset,
    wp::float32 var_radius,
    wp::int32 var_nface,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_point_out,
    wp::array_t<wp::int32> var_face_index_out,
    wp::array_t<wp::int32> var_group_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::float32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 2;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 3;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        const wp::int32 var_49 = 0;
        wp::int32 var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        const wp::int32 var_53 = 1;
        wp::int32 var_54;
        const wp::int32 var_55 = 2;
        wp::int32 var_56;
        const wp::int32 var_57 = 2;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        const wp::int32 var_61 = 2;
        wp::int32 var_62;
        wp::int32 var_63;
        const wp::int32 var_64 = 1;
        wp::int32 var_65;
        const wp::int32 var_66 = 3;
        wp::int32 var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        const wp::int32 var_70 = 0;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        const wp::int32 var_74 = 1;
        wp::int32 var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        const wp::int32 var_78 = 2;
        wp::int32 var_79;
        const wp::int32 var_80 = 0;
        wp::int32 var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        const wp::int32 var_86 = 1;
        wp::int32 var_87;
        const wp::int32 var_88 = 2;
        wp::int32 var_89;
        const wp::int32 var_90 = 2;
        wp::int32 var_91;
        //---------
        // forward
        // def _build_flex_2d_sides(                                                              <L 756>
        // worldid, shellid = wp.tid()                                                            <L 779>
        builtin_tid2d(var_0, var_1);
        // base = shell_adr + 2 * shellid                                                         <L 781>
        var_3 = wp::mul(var_2, var_1);
        var_4 = wp::add(var_shell_adr, var_3);
        // i0 = vert_adr + flex_shell[base + 0]                                                   <L 782>
        var_6 = wp::add(var_4, var_5);
        var_7 = wp::address(var_flex_shell, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::add(var_vert_adr, var_9);
        // i1 = vert_adr + flex_shell[base + 1]                                                   <L 783>
        var_11 = wp::add(var_4, var_10);
        var_12 = wp::address(var_flex_shell, var_11);
        var_14 = wp::load(var_12);
        var_13 = wp::add(var_vert_adr, var_14);
        // v0 = flexvert_xpos_in[worldid, i0]                                                     <L 785>
        var_15 = wp::address(var_flexvert_xpos_in, var_0, var_8);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // v1 = flexvert_xpos_in[worldid, i1]                                                     <L 786>
        var_18 = wp::address(var_flexvert_xpos_in, var_0, var_13);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // n0 = flexvert_norm_in[worldid, i0]                                                     <L 788>
        var_21 = wp::address(var_flexvert_norm_in, var_0, var_8);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // n1 = flexvert_norm_in[worldid, i1]                                                     <L 789>
        var_24 = wp::address(var_flexvert_norm_in, var_0, var_13);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // neg_radius = -radius                                                                   <L 791>
        var_27 = wp::neg(var_radius);
        // face_id0 = worldid * nface + face_offset + 2 * shellid                                 <L 794>
        var_28 = wp::mul(var_0, var_nface);
        var_29 = wp::add(var_28, var_face_offset);
        var_31 = wp::mul(var_30, var_1);
        var_32 = wp::add(var_29, var_31);
        // base0 = face_id0 * 3                                                                   <L 795>
        var_34 = wp::mul(var_32, var_33);
        // face_point_out[base0 + 0] = v0 + n0 * radius                                           <L 796>
        var_35 = wp::mul(var_22, var_radius);
        var_36 = wp::add(var_16, var_35);
        var_38 = wp::add(var_34, var_37);
        wp::array_store(var_face_point_out, var_38, var_36);
        // face_point_out[base0 + 1] = v1 + n1 * neg_radius                                       <L 797>
        var_39 = wp::mul(var_25, var_27);
        var_40 = wp::add(var_19, var_39);
        var_42 = wp::add(var_34, var_41);
        wp::array_store(var_face_point_out, var_42, var_40);
        // face_point_out[base0 + 2] = v1 + n1 * radius                                           <L 798>
        var_43 = wp::mul(var_25, var_radius);
        var_44 = wp::add(var_19, var_43);
        var_46 = wp::add(var_34, var_45);
        wp::array_store(var_face_point_out, var_46, var_44);
        // face_index_out[base0 + 0] = base0 + 0                                                  <L 799>
        var_48 = wp::add(var_34, var_47);
        var_50 = wp::add(var_34, var_49);
        wp::array_store(var_face_index_out, var_50, var_48);
        // face_index_out[base0 + 1] = base0 + 1                                                  <L 800>
        var_52 = wp::add(var_34, var_51);
        var_54 = wp::add(var_34, var_53);
        wp::array_store(var_face_index_out, var_54, var_52);
        // face_index_out[base0 + 2] = base0 + 2                                                  <L 801>
        var_56 = wp::add(var_34, var_55);
        var_58 = wp::add(var_34, var_57);
        wp::array_store(var_face_index_out, var_58, var_56);
        // face_id1 = worldid * nface + face_offset + 2 * shellid + 1                             <L 804>
        var_59 = wp::mul(var_0, var_nface);
        var_60 = wp::add(var_59, var_face_offset);
        var_62 = wp::mul(var_61, var_1);
        var_63 = wp::add(var_60, var_62);
        var_65 = wp::add(var_63, var_64);
        // base1 = face_id1 * 3                                                                   <L 805>
        var_67 = wp::mul(var_65, var_66);
        // face_point_out[base1 + 0] = v1 + n1 * neg_radius                                       <L 806>
        var_68 = wp::mul(var_25, var_27);
        var_69 = wp::add(var_19, var_68);
        var_71 = wp::add(var_67, var_70);
        wp::array_store(var_face_point_out, var_71, var_69);
        // face_point_out[base1 + 1] = v0 + n0 * neg_radius                                       <L 807>
        var_72 = wp::mul(var_22, var_27);
        var_73 = wp::add(var_16, var_72);
        var_75 = wp::add(var_67, var_74);
        wp::array_store(var_face_point_out, var_75, var_73);
        // face_point_out[base1 + 2] = v0 + n0 * radius                                           <L 808>
        var_76 = wp::mul(var_22, var_radius);
        var_77 = wp::add(var_16, var_76);
        var_79 = wp::add(var_67, var_78);
        wp::array_store(var_face_point_out, var_79, var_77);
        // face_index_out[base1 + 0] = base1 + 0                                                  <L 809>
        var_81 = wp::add(var_67, var_80);
        var_83 = wp::add(var_67, var_82);
        wp::array_store(var_face_index_out, var_83, var_81);
        // face_index_out[base1 + 1] = base1 + 1                                                  <L 810>
        var_85 = wp::add(var_67, var_84);
        var_87 = wp::add(var_67, var_86);
        wp::array_store(var_face_index_out, var_87, var_85);
        // face_index_out[base1 + 2] = base1 + 2                                                  <L 811>
        var_89 = wp::add(var_67, var_88);
        var_91 = wp::add(var_67, var_90);
        wp::array_store(var_face_index_out, var_91, var_89);
        // group_out[face_id0] = worldid                                                          <L 813>
        wp::array_store(var_group_out, var_32, var_0);
        // group_out[face_id1] = worldid                                                          <L 814>
        wp::array_store(var_group_out, var_65, var_0);
}



extern "C" __global__ void compute_bvh_group_roots_82f80f3b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::uint64 var_bvh_id,
    wp::array_t<wp::int32> var_group_root_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        //---------
        // forward
        // def compute_bvh_group_roots(                                                           <L 241>
        // tid = wp.tid()                                                                         <L 247>
        var_0 = builtin_tid1d();
        // root = wp.bvh_get_group_root(bvh_id, tid)                                              <L 248>
        var_1 = wp::bvh_get_group_root(var_bvh_id, var_0);
        // group_root_out[tid] = root                                                             <L 249>
        wp::array_store(var_group_root_out, var_0, var_1);
}



extern "C" __global__ void _update_flex_3d_face_points_97c40617_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_shelldataadr,
    wp::array_t<wp::int32> var_flex_shell,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::int32 var_flex_id,
    wp::int32 var_nface,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_point_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
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
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 3;
        wp::int32 var_11;
        const wp::int32 var_12 = 3;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::vec_t<3, wp::float32>* var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32>* var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32>* var_38;
        const wp::int32 var_39 = 2;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        //---------
        // forward
        // def _update_flex_3d_face_points(                                                       <L 963>
        // worldid, shellid = wp.tid()                                                            <L 976>
        builtin_tid2d(var_0, var_1);
        // shell_adr = flex_shelldataadr[flex_id]                                                 <L 978>
        var_2 = wp::address(var_flex_shelldataadr, var_flex_id);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // vert_adr = flex_vertadr[flex_id]                                                       <L 979>
        var_5 = wp::address(var_flex_vertadr, var_flex_id);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // face_id = worldid * nface + shellid                                                    <L 981>
        var_8 = wp::mul(var_0, var_nface);
        var_9 = wp::add(var_8, var_1);
        // fbase = face_id * 3                                                                    <L 982>
        var_11 = wp::mul(var_9, var_10);
        // sbase = shell_adr + shellid * 3                                                        <L 984>
        var_13 = wp::mul(var_1, var_12);
        var_14 = wp::add(var_3, var_13);
        // i0 = vert_adr + flex_shell[sbase + 0]                                                  <L 985>
        var_16 = wp::add(var_14, var_15);
        var_17 = wp::address(var_flex_shell, var_16);
        var_19 = wp::load(var_17);
        var_18 = wp::add(var_6, var_19);
        // i1 = vert_adr + flex_shell[sbase + 1]                                                  <L 986>
        var_21 = wp::add(var_14, var_20);
        var_22 = wp::address(var_flex_shell, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::add(var_6, var_24);
        // i2 = vert_adr + flex_shell[sbase + 2]                                                  <L 987>
        var_26 = wp::add(var_14, var_25);
        var_27 = wp::address(var_flex_shell, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::add(var_6, var_29);
        // face_point_out[fbase + 0] = flexvert_xpos_in[worldid, i0]                              <L 989>
        var_30 = wp::address(var_flexvert_xpos_in, var_0, var_18);
        var_32 = wp::add(var_11, var_31);
        var_33 = wp::load(var_30);
        wp::array_store(var_face_point_out, var_32, var_33);
        // face_point_out[fbase + 1] = flexvert_xpos_in[worldid, i1]                              <L 990>
        var_34 = wp::address(var_flexvert_xpos_in, var_0, var_23);
        var_36 = wp::add(var_11, var_35);
        var_37 = wp::load(var_34);
        wp::array_store(var_face_point_out, var_36, var_37);
        // face_point_out[fbase + 2] = flexvert_xpos_in[worldid, i2]                              <L 991>
        var_38 = wp::address(var_flexvert_xpos_in, var_0, var_28);
        var_40 = wp::add(var_11, var_39);
        var_41 = wp::load(var_38);
        wp::array_store(var_face_point_out, var_40, var_41);
}



extern "C" __global__ void _compute_bvh_bounds_a4bfd372_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_bvh_ngeom,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_bounds_size,
    wp::array_t<wp::vec_t<3, wp::float32>> var_hfield_bounds_size,
    wp::array_t<wp::vec_t<3, wp::float32>> var_lower_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_upper_out,
    wp::array_t<wp::int32> var_group_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
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
        wp::mat_t<3, 3, wp::float32>* var_8;
        wp::mat_t<3, 3, wp::float32> var_9;
        wp::mat_t<3, 3, wp::float32> var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 2;
        bool var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        const wp::int32 var_26 = 3;
        bool var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        const wp::int32 var_34 = 7;
        bool var_35;
        wp::shape_t* var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::shape_t var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        const wp::int32 var_56 = 4;
        bool var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        const wp::int32 var_60 = 5;
        bool var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 6;
        bool var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::int32 var_68 = 1;
        bool var_69;
        wp::shape_t* var_70;
        const wp::int32 var_71 = 0;
        wp::int32 var_72;
        wp::shape_t var_73;
        wp::int32 var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        wp::vec_t<3, wp::float32>* var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::slice_t var_83;
        const wp::int32 var_84 = 0;
        const wp::int32 var_85 = 3;
        const wp::int32 var_86 = 1;
        const wp::int32 var_87 = 2;
        wp::vec_t<3, wp::float32> var_88;
        const wp::int32 var_89 = 2;
        wp::float32 var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::int32 var_111;
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
        wp::int32 var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::int32 var_128;
        //---------
        // forward
        // def _compute_bvh_bounds(                                                               <L 178>
        // worldid, geom_local_id = wp.tid()                                                      <L 196>
        builtin_tid2d(var_0, var_1);
        // geom_id = enabled_geom_ids[geom_local_id]                                              <L 197>
        var_2 = wp::address(var_enabled_geom_ids, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pos = geom_xpos_in[worldid, geom_id]                                                   <L 199>
        var_5 = wp::address(var_geom_xpos_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // rot = geom_xmat_in[worldid, geom_id]                                                   <L 200>
        var_8 = wp::address(var_geom_xmat_in, var_0, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // size = geom_size[worldid % geom_size.shape[0], geom_id]                                <L 201>
        var_11 = &(var_geom_size.shape);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        var_15 = wp::mod(var_0, var_13);
        var_16 = wp::address(var_geom_size, var_15, var_3);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // type = geom_type[geom_id]                                                              <L 202>
        var_19 = wp::address(var_geom_type, var_3);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // if type == GeomType.SPHERE:                                                            <L 205>
        var_23 = (var_20 == var_22);
        if (var_23) {
            // lower_bound, upper_bound = _compute_sphere_bounds(pos, rot, size)                  <L 206>
            _compute_sphere_bounds_0(var_6, var_9, var_17, var_24, var_25);
        }
        if (!var_23) {
            // elif type == GeomType.CAPSULE:                                                     <L 207>
            var_27 = (var_20 == var_26);
            if (var_27) {
                // lower_bound, upper_bound = _compute_capsule_bounds(pos, rot, size)             <L 208>
                _compute_capsule_bounds_0(var_6, var_9, var_17, var_28, var_29);
            }
            if (!var_27) {
                // elif type == GeomType.PLANE:                                                   <L 209>
                var_31 = (var_20 == var_30);
                if (var_31) {
                    // lower_bound, upper_bound = _compute_plane_bounds(pos, rot, size)           <L 210>
                    _compute_plane_bounds_0(var_6, var_9, var_17, var_32, var_33);
                }
                if (!var_31) {
                    // elif type == GeomType.MESH:                                                <L 211>
                    var_35 = (var_20 == var_34);
                    if (var_35) {
                        // did = geom_dataid[worldid % geom_dataid.shape[0], geom_id]             <L 212>
                        var_36 = &(var_geom_dataid.shape);
                        var_39 = wp::load(var_36);
                        var_38 = wp::extract(var_39, var_37);
                        var_40 = wp::mod(var_0, var_38);
                        var_41 = wp::address(var_geom_dataid, var_40, var_3);
                        var_43 = wp::load(var_41);
                        var_42 = wp::copy(var_43);
                        // if did >= 0:                                                           <L 213>
                        var_45 = (var_42 >= var_44);
                        if (var_45) {
                            // size = mesh_bounds_size[did]                                       <L 214>
                            var_46 = wp::address(var_mesh_bounds_size, var_42);
                            var_48 = wp::load(var_46);
                            var_47 = wp::copy(var_48);
                            // lower_bound, upper_bound = _compute_box_bounds(pos, rot, size)       <L 215>
                            _compute_box_bounds_0(var_6, var_9, var_47, var_49, var_50);
                        }
                        if (!var_45) {
                            // lower_bound = pos                                                  <L 217>
                            var_51 = wp::copy(var_6);
                            // upper_bound = pos                                                  <L 218>
                            var_52 = wp::copy(var_6);
                        }
                        var_53 = wp::where(var_45, var_47, var_17);
                        var_54 = wp::where(var_45, var_49, var_51);
                        var_55 = wp::where(var_45, var_50, var_52);
                    }
                    if (!var_35) {
                        // elif type == GeomType.ELLIPSOID:                                       <L 219>
                        var_57 = (var_20 == var_56);
                        if (var_57) {
                            // lower_bound, upper_bound = _compute_ellipsoid_bounds(pos, rot, size)       <L 220>
                            _compute_ellipsoid_bounds_0(var_6, var_9, var_17, var_58, var_59);
                        }
                        if (!var_57) {
                            // elif type == GeomType.CYLINDER:                                    <L 221>
                            var_61 = (var_20 == var_60);
                            if (var_61) {
                                // lower_bound, upper_bound = _compute_cylinder_bounds(pos, rot, size)       <L 222>
                                _compute_cylinder_bounds_0(var_6, var_9, var_17, var_62, var_63);
                            }
                            if (!var_61) {
                                // elif type == GeomType.BOX:                                     <L 223>
                                var_65 = (var_20 == var_64);
                                if (var_65) {
                                    // lower_bound, upper_bound = _compute_box_bounds(pos, rot, size)       <L 224>
                                    _compute_box_bounds_0(var_6, var_9, var_17, var_66, var_67);
                                }
                                if (!var_65) {
                                    // elif type == GeomType.HFIELD:                              <L 225>
                                    var_69 = (var_20 == var_68);
                                    if (var_69) {
                                        // did = geom_dataid[worldid % geom_dataid.shape[0], geom_id]       <L 226>
                                        var_70 = &(var_geom_dataid.shape);
                                        var_73 = wp::load(var_70);
                                        var_72 = wp::extract(var_73, var_71);
                                        var_74 = wp::mod(var_0, var_72);
                                        var_75 = wp::address(var_geom_dataid, var_74, var_3);
                                        var_77 = wp::load(var_75);
                                        var_76 = wp::copy(var_77);
                                        // if did >= 0:                                           <L 227>
                                        var_79 = (var_76 >= var_78);
                                        if (var_79) {
                                            // size = hfield_bounds_size[did]                     <L 228>
                                            var_80 = wp::address(var_hfield_bounds_size, var_76);
                                            var_82 = wp::load(var_80);
                                            var_81 = wp::copy(var_82);
                                            // hfield_center = pos + rot[:, 2] * size[2]          <L 229>
                                            var_83 = wp::slice_t(var_84, var_85, var_86);
                                            var_88 = wp::extract<3>(var_9, var_83, var_87);
                                            var_90 = wp::extract(var_81, var_89);
                                            var_91 = wp::mul(var_88, var_90);
                                            var_92 = wp::add(var_6, var_91);
                                            // lower_bound, upper_bound = _compute_box_bounds(hfield_center, rot, size)       <L 230>
                                            _compute_box_bounds_0(var_92, var_9, var_81, var_93, var_94);
                                        }
                                        if (!var_79) {
                                            // lower_bound = pos                                  <L 232>
                                            var_95 = wp::copy(var_6);
                                            // upper_bound = pos                                  <L 233>
                                            var_96 = wp::copy(var_6);
                                        }
                                        var_97 = wp::where(var_79, var_81, var_17);
                                        var_98 = wp::where(var_79, var_93, var_95);
                                        var_99 = wp::where(var_79, var_94, var_96);
                                    }
                                    var_100 = wp::where(var_69, var_97, var_17);
                                }
                                var_101 = wp::where(var_65, var_17, var_100);
                                var_102 = wp::where(var_65, var_66, var_98);
                                var_103 = wp::where(var_65, var_67, var_99);
                            }
                            var_104 = wp::where(var_61, var_17, var_101);
                            var_105 = wp::where(var_61, var_62, var_102);
                            var_106 = wp::where(var_61, var_63, var_103);
                        }
                        var_107 = wp::where(var_57, var_17, var_104);
                        var_108 = wp::where(var_57, var_58, var_105);
                        var_109 = wp::where(var_57, var_59, var_106);
                    }
                    var_110 = wp::where(var_35, var_53, var_107);
                    var_111 = wp::where(var_35, var_42, var_76);
                    var_112 = wp::where(var_35, var_54, var_108);
                    var_113 = wp::where(var_35, var_55, var_109);
                }
                var_114 = wp::where(var_31, var_17, var_110);
                var_115 = wp::where(var_31, var_32, var_112);
                var_116 = wp::where(var_31, var_33, var_113);
            }
            var_117 = wp::where(var_27, var_17, var_114);
            var_118 = wp::where(var_27, var_28, var_115);
            var_119 = wp::where(var_27, var_29, var_116);
        }
        var_120 = wp::where(var_23, var_17, var_117);
        var_121 = wp::where(var_23, var_24, var_118);
        var_122 = wp::where(var_23, var_25, var_119);
        // lower_out[worldid * bvh_ngeom + geom_local_id] = lower_bound                           <L 235>
        var_123 = wp::mul(var_0, var_bvh_ngeom);
        var_124 = wp::add(var_123, var_1);
        wp::array_store(var_lower_out, var_124, var_121);
        // upper_out[worldid * bvh_ngeom + geom_local_id] = upper_bound                           <L 236>
        var_125 = wp::mul(var_0, var_bvh_ngeom);
        var_126 = wp::add(var_125, var_1);
        wp::array_store(var_upper_out, var_126, var_122);
        // group_out[worldid * bvh_ngeom + geom_local_id] = worldid                               <L 237>
        var_127 = wp::mul(var_0, var_bvh_ngeom);
        var_128 = wp::add(var_127, var_1);
        wp::array_store(var_group_out, var_128, var_0);
}



extern "C" __global__ void _build_flex_2d_elements_d1d4bb6d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_flex_elem,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_norm_in,
    wp::int32 var_elem_adr,
    wp::int32 var_vert_adr,
    wp::int32 var_face_offset,
    wp::float32 var_radius,
    wp::int32 var_nfaces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_point_out,
    wp::array_t<wp::int32> var_face_index_out,
    wp::array_t<wp::int32> var_group_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 3;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 2;
        wp::int32 var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 3;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        const wp::int32 var_61 = 2;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        wp::int32 var_64;
        const wp::int32 var_65 = 0;
        wp::int32 var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        const wp::int32 var_71 = 2;
        wp::int32 var_72;
        const wp::int32 var_73 = 2;
        wp::int32 var_74;
        wp::int32 var_75;
        const wp::int32 var_76 = 2;
        wp::int32 var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 3;
        wp::int32 var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::int32 var_86;
        const wp::int32 var_87 = 2;
        wp::int32 var_88;
        const wp::int32 var_89 = 0;
        wp::int32 var_90;
        const wp::int32 var_91 = 0;
        wp::int32 var_92;
        const wp::int32 var_93 = 2;
        wp::int32 var_94;
        const wp::int32 var_95 = 1;
        wp::int32 var_96;
        const wp::int32 var_97 = 1;
        wp::int32 var_98;
        const wp::int32 var_99 = 2;
        wp::int32 var_100;
        //---------
        // forward
        // def _build_flex_2d_elements(                                                           <L 682>
        // worldid, elemid = wp.tid()                                                             <L 703>
        builtin_tid2d(var_0, var_1);
        // base = elem_adr + elemid * 3                                                           <L 705>
        var_3 = wp::mul(var_1, var_2);
        var_4 = wp::add(var_elem_adr, var_3);
        // i0 = vert_adr + flex_elem[base + 0]                                                    <L 706>
        var_6 = wp::add(var_4, var_5);
        var_7 = wp::address(var_flex_elem, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::add(var_vert_adr, var_9);
        // i1 = vert_adr + flex_elem[base + 1]                                                    <L 707>
        var_11 = wp::add(var_4, var_10);
        var_12 = wp::address(var_flex_elem, var_11);
        var_14 = wp::load(var_12);
        var_13 = wp::add(var_vert_adr, var_14);
        // i2 = vert_adr + flex_elem[base + 2]                                                    <L 708>
        var_16 = wp::add(var_4, var_15);
        var_17 = wp::address(var_flex_elem, var_16);
        var_19 = wp::load(var_17);
        var_18 = wp::add(var_vert_adr, var_19);
        // v0 = flexvert_xpos_in[worldid, i0]                                                     <L 710>
        var_20 = wp::address(var_flexvert_xpos_in, var_0, var_8);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // v1 = flexvert_xpos_in[worldid, i1]                                                     <L 711>
        var_23 = wp::address(var_flexvert_xpos_in, var_0, var_13);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // v2 = flexvert_xpos_in[worldid, i2]                                                     <L 712>
        var_26 = wp::address(var_flexvert_xpos_in, var_0, var_18);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // n0 = flexvert_norm_in[worldid, i0]                                                     <L 714>
        var_29 = wp::address(var_flexvert_norm_in, var_0, var_8);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // n1 = flexvert_norm_in[worldid, i1]                                                     <L 715>
        var_32 = wp::address(var_flexvert_norm_in, var_0, var_13);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // n2 = flexvert_norm_in[worldid, i2]                                                     <L 716>
        var_35 = wp::address(var_flexvert_norm_in, var_0, var_18);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // p0_pos = v0 + radius * n0                                                              <L 718>
        var_38 = wp::mul(var_radius, var_30);
        var_39 = wp::add(var_21, var_38);
        // p1_pos = v1 + radius * n1                                                              <L 719>
        var_40 = wp::mul(var_radius, var_33);
        var_41 = wp::add(var_24, var_40);
        // p2_pos = v2 + radius * n2                                                              <L 720>
        var_42 = wp::mul(var_radius, var_36);
        var_43 = wp::add(var_27, var_42);
        // p0_neg = v0 - radius * n0                                                              <L 722>
        var_44 = wp::mul(var_radius, var_30);
        var_45 = wp::sub(var_21, var_44);
        // p1_neg = v1 - radius * n1                                                              <L 723>
        var_46 = wp::mul(var_radius, var_33);
        var_47 = wp::sub(var_24, var_46);
        // p2_neg = v2 - radius * n2                                                              <L 724>
        var_48 = wp::mul(var_radius, var_36);
        var_49 = wp::sub(var_27, var_48);
        // world_face_offset = worldid * nfaces                                                   <L 726>
        var_50 = wp::mul(var_0, var_nfaces);
        // face_id0 = world_face_offset + face_offset + 2 * elemid                                <L 729>
        var_51 = wp::add(var_50, var_face_offset);
        var_53 = wp::mul(var_52, var_1);
        var_54 = wp::add(var_51, var_53);
        // base0 = face_id0 * 3                                                                   <L 730>
        var_56 = wp::mul(var_54, var_55);
        // face_point_out[base0 + 0] = p0_pos                                                     <L 731>
        var_58 = wp::add(var_56, var_57);
        wp::array_store(var_face_point_out, var_58, var_39);
        // face_point_out[base0 + 1] = p1_pos                                                     <L 732>
        var_60 = wp::add(var_56, var_59);
        wp::array_store(var_face_point_out, var_60, var_41);
        // face_point_out[base0 + 2] = p2_pos                                                     <L 733>
        var_62 = wp::add(var_56, var_61);
        wp::array_store(var_face_point_out, var_62, var_43);
        // face_index_out[base0 + 0] = base0 + 0                                                  <L 735>
        var_64 = wp::add(var_56, var_63);
        var_66 = wp::add(var_56, var_65);
        wp::array_store(var_face_index_out, var_66, var_64);
        // face_index_out[base0 + 1] = base0 + 1                                                  <L 736>
        var_68 = wp::add(var_56, var_67);
        var_70 = wp::add(var_56, var_69);
        wp::array_store(var_face_index_out, var_70, var_68);
        // face_index_out[base0 + 2] = base0 + 2                                                  <L 737>
        var_72 = wp::add(var_56, var_71);
        var_74 = wp::add(var_56, var_73);
        wp::array_store(var_face_index_out, var_74, var_72);
        // group_out[face_id0] = worldid                                                          <L 739>
        wp::array_store(var_group_out, var_54, var_0);
        // face_id1 = world_face_offset + face_offset + 2 * elemid + 1                            <L 742>
        var_75 = wp::add(var_50, var_face_offset);
        var_77 = wp::mul(var_76, var_1);
        var_78 = wp::add(var_75, var_77);
        var_80 = wp::add(var_78, var_79);
        // base1 = face_id1 * 3                                                                   <L 743>
        var_82 = wp::mul(var_80, var_81);
        // face_point_out[base1 + 0] = p0_neg                                                     <L 744>
        var_84 = wp::add(var_82, var_83);
        wp::array_store(var_face_point_out, var_84, var_45);
        // face_point_out[base1 + 1] = p1_neg                                                     <L 745>
        var_86 = wp::add(var_82, var_85);
        wp::array_store(var_face_point_out, var_86, var_47);
        // face_point_out[base1 + 2] = p2_neg                                                     <L 746>
        var_88 = wp::add(var_82, var_87);
        wp::array_store(var_face_point_out, var_88, var_49);
        // face_index_out[base1 + 0] = base1 + 0                                                  <L 748>
        var_90 = wp::add(var_82, var_89);
        var_92 = wp::add(var_82, var_91);
        wp::array_store(var_face_index_out, var_92, var_90);
        // face_index_out[base1 + 1] = base1 + 2                                                  <L 749>
        var_94 = wp::add(var_82, var_93);
        var_96 = wp::add(var_82, var_95);
        wp::array_store(var_face_index_out, var_96, var_94);
        // face_index_out[base1 + 2] = base1 + 1                                                  <L 750>
        var_98 = wp::add(var_82, var_97);
        var_100 = wp::add(var_82, var_99);
        wp::array_store(var_face_index_out, var_100, var_98);
        // group_out[face_id1] = worldid                                                          <L 752>
        wp::array_store(var_group_out, var_80, var_0);
}



extern "C" __global__ void _compute_splat_bounds_dc069f2a_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_splat_position,
    wp::array_t<wp::quat_t<wp::float32>> var_splat_rotation,
    wp::array_t<wp::vec_t<3, wp::float32>> var_splat_scale,
    wp::array_t<wp::vec_t<4, wp::float32>> var_splat_rgba,
    wp::array_t<wp::vec_t<3, wp::float32>> var_lower_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_upper_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3, wp::float32>* var_1;
        wp::vec_t<3, wp::float32> var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::quat_t<wp::float32>* var_4;
        wp::quat_t<wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<4, wp::float32>* var_10;
        const wp::int32 var_11 = 3;
        wp::float32 var_12;
        wp::vec_t<4, wp::float32> var_13;
        const wp::float32 var_14 = 1e-06;
        wp::float32 var_15;
        const wp::float32 var_16 = 0.01;
        wp::float32 var_17;
        const wp::float32 var_18 = 1e-06;
        const wp::float32 var_19 = 0.97;
        wp::float32 var_20;
        const wp::float32 var_21 = -2.0;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::float32 var_25 = 10000000000.0;
        wp::vec_t<3, wp::float32> var_26;
        const wp::float32 var_27 = -10000000000.0;
        wp::vec_t<3, wp::float32> var_28;
        const wp::int32 var_29 = 0;
        const wp::int32 var_30 = 0;
        const wp::int32 var_31 = 0;
        const wp::int32 var_32 = 0;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 2.0;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::float32 var_38 = 1.0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 2.0;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::float32 var_47 = 1.0;
        wp::float32 var_48;
        wp::float32 var_49;
        const wp::int32 var_50 = 2;
        wp::float32 var_51;
        wp::float32 var_52;
        const wp::float32 var_53 = 2.0;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 1.0;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 1;
        const wp::int32 var_65 = 0;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 2.0;
        wp::float32 var_69;
        wp::float32 var_70;
        const wp::float32 var_71 = 1.0;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 1;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::float32 var_77 = 2.0;
        wp::float32 var_78;
        wp::float32 var_79;
        const wp::float32 var_80 = 1.0;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 2;
        wp::float32 var_84;
        wp::float32 var_85;
        const wp::float32 var_86 = 2.0;
        wp::float32 var_87;
        wp::float32 var_88;
        const wp::float32 var_89 = 1.0;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        const wp::int32 var_97 = 1;
        const wp::int32 var_98 = 0;
        const wp::int32 var_99 = 0;
        wp::float32 var_100;
        wp::float32 var_101;
        const wp::float32 var_102 = 2.0;
        wp::float32 var_103;
        wp::float32 var_104;
        const wp::float32 var_105 = 1.0;
        wp::float32 var_106;
        wp::float32 var_107;
        const wp::int32 var_108 = 1;
        wp::float32 var_109;
        wp::float32 var_110;
        const wp::float32 var_111 = 2.0;
        wp::float32 var_112;
        wp::float32 var_113;
        const wp::float32 var_114 = 1.0;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 2;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::float32 var_120 = 2.0;
        wp::float32 var_121;
        wp::float32 var_122;
        const wp::float32 var_123 = 1.0;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        const wp::int32 var_131 = 1;
        const wp::int32 var_132 = 0;
        wp::float32 var_133;
        wp::float32 var_134;
        const wp::float32 var_135 = 2.0;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::float32 var_138 = 1.0;
        wp::float32 var_139;
        wp::float32 var_140;
        const wp::int32 var_141 = 1;
        wp::float32 var_142;
        wp::float32 var_143;
        const wp::float32 var_144 = 2.0;
        wp::float32 var_145;
        wp::float32 var_146;
        const wp::float32 var_147 = 1.0;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 2;
        wp::float32 var_151;
        wp::float32 var_152;
        const wp::float32 var_153 = 2.0;
        wp::float32 var_154;
        wp::float32 var_155;
        const wp::float32 var_156 = 1.0;
        wp::float32 var_157;
        wp::float32 var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::vec_t<3, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::vec_t<3, wp::float32> var_163;
        const wp::int32 var_164 = 1;
        const wp::int32 var_165 = 0;
        const wp::int32 var_166 = 0;
        const wp::int32 var_167 = 0;
        wp::float32 var_168;
        wp::float32 var_169;
        const wp::float32 var_170 = 2.0;
        wp::float32 var_171;
        wp::float32 var_172;
        const wp::float32 var_173 = 1.0;
        wp::float32 var_174;
        wp::float32 var_175;
        const wp::int32 var_176 = 1;
        wp::float32 var_177;
        wp::float32 var_178;
        const wp::float32 var_179 = 2.0;
        wp::float32 var_180;
        wp::float32 var_181;
        const wp::float32 var_182 = 1.0;
        wp::float32 var_183;
        wp::float32 var_184;
        const wp::int32 var_185 = 2;
        wp::float32 var_186;
        wp::float32 var_187;
        const wp::float32 var_188 = 2.0;
        wp::float32 var_189;
        wp::float32 var_190;
        const wp::float32 var_191 = 1.0;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::vec_t<3, wp::float32> var_194;
        wp::vec_t<3, wp::float32> var_195;
        wp::vec_t<3, wp::float32> var_196;
        wp::vec_t<3, wp::float32> var_197;
        wp::vec_t<3, wp::float32> var_198;
        const wp::int32 var_199 = 1;
        const wp::int32 var_200 = 0;
        wp::float32 var_201;
        wp::float32 var_202;
        const wp::float32 var_203 = 2.0;
        wp::float32 var_204;
        wp::float32 var_205;
        const wp::float32 var_206 = 1.0;
        wp::float32 var_207;
        wp::float32 var_208;
        const wp::int32 var_209 = 1;
        wp::float32 var_210;
        wp::float32 var_211;
        const wp::float32 var_212 = 2.0;
        wp::float32 var_213;
        wp::float32 var_214;
        const wp::float32 var_215 = 1.0;
        wp::float32 var_216;
        wp::float32 var_217;
        const wp::int32 var_218 = 2;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::float32 var_221 = 2.0;
        wp::float32 var_222;
        wp::float32 var_223;
        const wp::float32 var_224 = 1.0;
        wp::float32 var_225;
        wp::float32 var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::vec_t<3, wp::float32> var_231;
        const wp::int32 var_232 = 1;
        const wp::int32 var_233 = 0;
        const wp::int32 var_234 = 0;
        wp::float32 var_235;
        wp::float32 var_236;
        const wp::float32 var_237 = 2.0;
        wp::float32 var_238;
        wp::float32 var_239;
        const wp::float32 var_240 = 1.0;
        wp::float32 var_241;
        wp::float32 var_242;
        const wp::int32 var_243 = 1;
        wp::float32 var_244;
        wp::float32 var_245;
        const wp::float32 var_246 = 2.0;
        wp::float32 var_247;
        wp::float32 var_248;
        const wp::float32 var_249 = 1.0;
        wp::float32 var_250;
        wp::float32 var_251;
        const wp::int32 var_252 = 2;
        wp::float32 var_253;
        wp::float32 var_254;
        const wp::float32 var_255 = 2.0;
        wp::float32 var_256;
        wp::float32 var_257;
        const wp::float32 var_258 = 1.0;
        wp::float32 var_259;
        wp::float32 var_260;
        wp::vec_t<3, wp::float32> var_261;
        wp::vec_t<3, wp::float32> var_262;
        wp::vec_t<3, wp::float32> var_263;
        wp::vec_t<3, wp::float32> var_264;
        wp::vec_t<3, wp::float32> var_265;
        const wp::int32 var_266 = 1;
        const wp::int32 var_267 = 0;
        wp::float32 var_268;
        wp::float32 var_269;
        const wp::float32 var_270 = 2.0;
        wp::float32 var_271;
        wp::float32 var_272;
        const wp::float32 var_273 = 1.0;
        wp::float32 var_274;
        wp::float32 var_275;
        const wp::int32 var_276 = 1;
        wp::float32 var_277;
        wp::float32 var_278;
        const wp::float32 var_279 = 2.0;
        wp::float32 var_280;
        wp::float32 var_281;
        const wp::float32 var_282 = 1.0;
        wp::float32 var_283;
        wp::float32 var_284;
        const wp::int32 var_285 = 2;
        wp::float32 var_286;
        wp::float32 var_287;
        const wp::float32 var_288 = 2.0;
        wp::float32 var_289;
        wp::float32 var_290;
        const wp::float32 var_291 = 1.0;
        wp::float32 var_292;
        wp::float32 var_293;
        wp::vec_t<3, wp::float32> var_294;
        wp::vec_t<3, wp::float32> var_295;
        wp::vec_t<3, wp::float32> var_296;
        wp::vec_t<3, wp::float32> var_297;
        wp::vec_t<3, wp::float32> var_298;
        //---------
        // forward
        // def _compute_splat_bounds(                                                             <L 1183>
        // tid = wp.tid()                                                                         <L 1193>
        var_0 = builtin_tid1d();
        // pos = splat_position[tid]                                                              <L 1194>
        var_1 = wp::address(var_splat_position, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // rot = splat_rotation[tid]                                                              <L 1195>
        var_4 = wp::address(var_splat_rotation, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // scale = splat_scale[tid]                                                               <L 1196>
        var_7 = wp::address(var_splat_scale, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // opacity = wp.max(splat_rgba[tid][3], 1.0e-6)                                           <L 1197>
        var_10 = wp::address(var_splat_rgba, var_0);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_15 = wp::max(var_12, var_14);
        // response = wp.clamp(wp.static(SPLAT_MIN_RESPONSE) / opacity, 1.0e-6, 0.97)             <L 1200>
        var_17 = wp::div(var_16, var_15);
        var_20 = wp::clamp(var_17, var_18, var_19);
        // radius = wp.sqrt(-2.0 * wp.log(response))                                              <L 1201>
        var_22 = wp::log(var_20);
        var_23 = wp::mul(var_21, var_22);
        var_24 = wp::sqrt(var_23);
        // lo = wp.vec3(MJ_MAXVAL)                                                                <L 1203>
        var_26 = wp::vec_t<3, wp::float32>(var_25);
        // hi = wp.vec3(-MJ_MAXVAL)                                                               <L 1204>
        var_28 = wp::vec_t<3, wp::float32>(var_27);
        // for x in range(2):                                                                     <L 1205>
        // for y in range(2):                                                                     <L 1206>
        // for z in range(2):                                                                     <L 1207>
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_33 = wp::extract(var_8, var_32);
        var_34 = wp::mul(var_33, var_24);
        var_36 = wp::float(var_29);
        var_37 = wp::mul(var_35, var_36);
        var_39 = wp::sub(var_37, var_38);
        var_40 = wp::mul(var_34, var_39);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_42 = wp::extract(var_8, var_41);
        var_43 = wp::mul(var_42, var_24);
        var_45 = wp::float(var_30);
        var_46 = wp::mul(var_44, var_45);
        var_48 = wp::sub(var_46, var_47);
        var_49 = wp::mul(var_43, var_48);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_51 = wp::extract(var_8, var_50);
        var_52 = wp::mul(var_51, var_24);
        var_54 = wp::float(var_31);
        var_55 = wp::mul(var_53, var_54);
        var_57 = wp::sub(var_55, var_56);
        var_58 = wp::mul(var_52, var_57);
        var_59 = wp::vec_t<3, wp::float32>(var_40, var_49, var_58);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_60 = rot_vec_quat_0(var_59, var_5);
        var_61 = wp::add(var_2, var_60);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_62 = wp::min(var_26, var_61);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_63 = wp::max(var_28, var_61);
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_66 = wp::extract(var_8, var_65);
        var_67 = wp::mul(var_66, var_24);
        var_69 = wp::float(var_29);
        var_70 = wp::mul(var_68, var_69);
        var_72 = wp::sub(var_70, var_71);
        var_73 = wp::mul(var_67, var_72);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_75 = wp::extract(var_8, var_74);
        var_76 = wp::mul(var_75, var_24);
        var_78 = wp::float(var_30);
        var_79 = wp::mul(var_77, var_78);
        var_81 = wp::sub(var_79, var_80);
        var_82 = wp::mul(var_76, var_81);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_84 = wp::extract(var_8, var_83);
        var_85 = wp::mul(var_84, var_24);
        var_87 = wp::float(var_64);
        var_88 = wp::mul(var_86, var_87);
        var_90 = wp::sub(var_88, var_89);
        var_91 = wp::mul(var_85, var_90);
        var_92 = wp::vec_t<3, wp::float32>(var_73, var_82, var_91);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_93 = rot_vec_quat_0(var_92, var_5);
        var_94 = wp::add(var_2, var_93);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_95 = wp::min(var_62, var_94);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_96 = wp::max(var_63, var_94);
        // for z in range(2):                                                                     <L 1207>
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_100 = wp::extract(var_8, var_99);
        var_101 = wp::mul(var_100, var_24);
        var_103 = wp::float(var_29);
        var_104 = wp::mul(var_102, var_103);
        var_106 = wp::sub(var_104, var_105);
        var_107 = wp::mul(var_101, var_106);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_109 = wp::extract(var_8, var_108);
        var_110 = wp::mul(var_109, var_24);
        var_112 = wp::float(var_97);
        var_113 = wp::mul(var_111, var_112);
        var_115 = wp::sub(var_113, var_114);
        var_116 = wp::mul(var_110, var_115);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_118 = wp::extract(var_8, var_117);
        var_119 = wp::mul(var_118, var_24);
        var_121 = wp::float(var_98);
        var_122 = wp::mul(var_120, var_121);
        var_124 = wp::sub(var_122, var_123);
        var_125 = wp::mul(var_119, var_124);
        var_126 = wp::vec_t<3, wp::float32>(var_107, var_116, var_125);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_127 = rot_vec_quat_0(var_126, var_5);
        var_128 = wp::add(var_2, var_127);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_129 = wp::min(var_95, var_128);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_130 = wp::max(var_96, var_128);
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_133 = wp::extract(var_8, var_132);
        var_134 = wp::mul(var_133, var_24);
        var_136 = wp::float(var_29);
        var_137 = wp::mul(var_135, var_136);
        var_139 = wp::sub(var_137, var_138);
        var_140 = wp::mul(var_134, var_139);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_142 = wp::extract(var_8, var_141);
        var_143 = wp::mul(var_142, var_24);
        var_145 = wp::float(var_97);
        var_146 = wp::mul(var_144, var_145);
        var_148 = wp::sub(var_146, var_147);
        var_149 = wp::mul(var_143, var_148);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_151 = wp::extract(var_8, var_150);
        var_152 = wp::mul(var_151, var_24);
        var_154 = wp::float(var_131);
        var_155 = wp::mul(var_153, var_154);
        var_157 = wp::sub(var_155, var_156);
        var_158 = wp::mul(var_152, var_157);
        var_159 = wp::vec_t<3, wp::float32>(var_140, var_149, var_158);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_160 = rot_vec_quat_0(var_159, var_5);
        var_161 = wp::add(var_2, var_160);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_162 = wp::min(var_129, var_161);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_163 = wp::max(var_130, var_161);
        // for y in range(2):                                                                     <L 1206>
        // for z in range(2):                                                                     <L 1207>
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_168 = wp::extract(var_8, var_167);
        var_169 = wp::mul(var_168, var_24);
        var_171 = wp::float(var_164);
        var_172 = wp::mul(var_170, var_171);
        var_174 = wp::sub(var_172, var_173);
        var_175 = wp::mul(var_169, var_174);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_177 = wp::extract(var_8, var_176);
        var_178 = wp::mul(var_177, var_24);
        var_180 = wp::float(var_165);
        var_181 = wp::mul(var_179, var_180);
        var_183 = wp::sub(var_181, var_182);
        var_184 = wp::mul(var_178, var_183);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_186 = wp::extract(var_8, var_185);
        var_187 = wp::mul(var_186, var_24);
        var_189 = wp::float(var_166);
        var_190 = wp::mul(var_188, var_189);
        var_192 = wp::sub(var_190, var_191);
        var_193 = wp::mul(var_187, var_192);
        var_194 = wp::vec_t<3, wp::float32>(var_175, var_184, var_193);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_195 = rot_vec_quat_0(var_194, var_5);
        var_196 = wp::add(var_2, var_195);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_197 = wp::min(var_162, var_196);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_198 = wp::max(var_163, var_196);
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_201 = wp::extract(var_8, var_200);
        var_202 = wp::mul(var_201, var_24);
        var_204 = wp::float(var_164);
        var_205 = wp::mul(var_203, var_204);
        var_207 = wp::sub(var_205, var_206);
        var_208 = wp::mul(var_202, var_207);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_210 = wp::extract(var_8, var_209);
        var_211 = wp::mul(var_210, var_24);
        var_213 = wp::float(var_165);
        var_214 = wp::mul(var_212, var_213);
        var_216 = wp::sub(var_214, var_215);
        var_217 = wp::mul(var_211, var_216);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_219 = wp::extract(var_8, var_218);
        var_220 = wp::mul(var_219, var_24);
        var_222 = wp::float(var_199);
        var_223 = wp::mul(var_221, var_222);
        var_225 = wp::sub(var_223, var_224);
        var_226 = wp::mul(var_220, var_225);
        var_227 = wp::vec_t<3, wp::float32>(var_208, var_217, var_226);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_228 = rot_vec_quat_0(var_227, var_5);
        var_229 = wp::add(var_2, var_228);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_230 = wp::min(var_197, var_229);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_231 = wp::max(var_198, var_229);
        // for z in range(2):                                                                     <L 1207>
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_235 = wp::extract(var_8, var_234);
        var_236 = wp::mul(var_235, var_24);
        var_238 = wp::float(var_164);
        var_239 = wp::mul(var_237, var_238);
        var_241 = wp::sub(var_239, var_240);
        var_242 = wp::mul(var_236, var_241);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_244 = wp::extract(var_8, var_243);
        var_245 = wp::mul(var_244, var_24);
        var_247 = wp::float(var_232);
        var_248 = wp::mul(var_246, var_247);
        var_250 = wp::sub(var_248, var_249);
        var_251 = wp::mul(var_245, var_250);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_253 = wp::extract(var_8, var_252);
        var_254 = wp::mul(var_253, var_24);
        var_256 = wp::float(var_233);
        var_257 = wp::mul(var_255, var_256);
        var_259 = wp::sub(var_257, var_258);
        var_260 = wp::mul(var_254, var_259);
        var_261 = wp::vec_t<3, wp::float32>(var_242, var_251, var_260);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_262 = rot_vec_quat_0(var_261, var_5);
        var_263 = wp::add(var_2, var_262);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_264 = wp::min(var_230, var_263);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_265 = wp::max(var_231, var_263);
        // corner = wp.vec3(                                                                      <L 1208>
        // scale[0] * radius * (2.0 * float(x) - 1.0),                                            <L 1209>
        var_268 = wp::extract(var_8, var_267);
        var_269 = wp::mul(var_268, var_24);
        var_271 = wp::float(var_164);
        var_272 = wp::mul(var_270, var_271);
        var_274 = wp::sub(var_272, var_273);
        var_275 = wp::mul(var_269, var_274);
        // scale[1] * radius * (2.0 * float(y) - 1.0),                                            <L 1210>
        var_277 = wp::extract(var_8, var_276);
        var_278 = wp::mul(var_277, var_24);
        var_280 = wp::float(var_232);
        var_281 = wp::mul(var_279, var_280);
        var_283 = wp::sub(var_281, var_282);
        var_284 = wp::mul(var_278, var_283);
        // scale[2] * radius * (2.0 * float(z) - 1.0),                                            <L 1211>
        var_286 = wp::extract(var_8, var_285);
        var_287 = wp::mul(var_286, var_24);
        var_289 = wp::float(var_266);
        var_290 = wp::mul(var_288, var_289);
        var_292 = wp::sub(var_290, var_291);
        var_293 = wp::mul(var_287, var_292);
        var_294 = wp::vec_t<3, wp::float32>(var_275, var_284, var_293);
        // point = pos + mjmath.rot_vec_quat(corner, rot)                                         <L 1213>
        var_295 = rot_vec_quat_0(var_294, var_5);
        var_296 = wp::add(var_2, var_295);
        // lo = wp.min(lo, point)                                                                 <L 1214>
        var_297 = wp::min(var_264, var_296);
        // hi = wp.max(hi, point)                                                                 <L 1215>
        var_298 = wp::max(var_265, var_296);
        // lower_out[tid] = lo                                                                    <L 1217>
        wp::array_store(var_lower_out, var_0, var_297);
        // upper_out[tid] = hi                                                                    <L 1218>
        wp::array_store(var_upper_out, var_0, var_298);
}



extern "C" __global__ void accumulate_flex_vertex_normals_dcfa7faf_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_elemadr,
    wp::array_t<wp::int32> var_flex_elemnum,
    wp::array_t<wp::int32> var_flex_elemdataadr,
    wp::array_t<wp::int32> var_flex_elem,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_norm_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
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
        bool var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        const wp::int32 var_20 = 3;
        bool var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 3;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32>* var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        //---------
        // forward
        // def accumulate_flex_vertex_normals(                                                    <L 626>
        // worldid, elemid = wp.tid()                                                             <L 641>
        builtin_tid2d(var_0, var_1);
        // for i in range(nflex):                                                                 <L 643>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = elemid - flex_elemadr[i]                                                   <L 644>
            var_4 = wp::address(var_flex_elemadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_elemnum[i]:                                         <L 645>
            var_9 = (var_5 >= var_8);
            var_7 = var_9;
            if (var_7) {
                var_10 = wp::address(var_flex_elemnum, var_3);
                var_12 = wp::load(var_10);
                var_11 = (var_5 < var_12);
                var_7 = var_7 && var_11;
            }
            if (var_7) {
                // f = i                                                                          <L 646>
                var_13 = wp::copy(var_3);
                // break                                                                          <L 647>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // if flex_dim[f] == 1 or flex_dim[f] == 3:                                               <L 649>
        var_15 = wp::address(var_flex_dim, var_13);
        var_18 = wp::load(var_15);
        var_17 = (var_18 == var_16);
        var_14 = var_17;
        if (!var_14) {
            var_19 = wp::address(var_flex_dim, var_13);
            var_22 = wp::load(var_19);
            var_21 = (var_22 == var_20);
            var_14 = var_14 || var_21;
        }
        if (var_14) {
            // return                                                                             <L 650>
            return;
        }
        // local_elemid = elemid - flex_elemadr[f]                                                <L 652>
        var_23 = wp::address(var_flex_elemadr, var_13);
        var_25 = wp::load(var_23);
        var_24 = wp::sub(var_1, var_25);
        // elem_adr = flex_elemdataadr[f]                                                         <L 653>
        var_26 = wp::address(var_flex_elemdataadr, var_13);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // vert_adr = flex_vertadr[f]                                                             <L 654>
        var_29 = wp::address(var_flex_vertadr, var_13);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // elem_base = elem_adr + local_elemid * 3                                                <L 655>
        var_33 = wp::mul(var_24, var_32);
        var_34 = wp::add(var_27, var_33);
        // i0 = vert_adr + flex_elem[elem_base + 0]                                               <L 656>
        var_36 = wp::add(var_34, var_35);
        var_37 = wp::address(var_flex_elem, var_36);
        var_39 = wp::load(var_37);
        var_38 = wp::add(var_30, var_39);
        // i1 = vert_adr + flex_elem[elem_base + 1]                                               <L 657>
        var_41 = wp::add(var_34, var_40);
        var_42 = wp::address(var_flex_elem, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::add(var_30, var_44);
        // i2 = vert_adr + flex_elem[elem_base + 2]                                               <L 658>
        var_46 = wp::add(var_34, var_45);
        var_47 = wp::address(var_flex_elem, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::add(var_30, var_49);
        // v0 = flexvert_xpos_in[worldid, i0]                                                     <L 660>
        var_50 = wp::address(var_flexvert_xpos_in, var_0, var_38);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // v1 = flexvert_xpos_in[worldid, i1]                                                     <L 661>
        var_53 = wp::address(var_flexvert_xpos_in, var_0, var_43);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // v2 = flexvert_xpos_in[worldid, i2]                                                     <L 662>
        var_56 = wp::address(var_flexvert_xpos_in, var_0, var_48);
        var_58 = wp::load(var_56);
        var_57 = wp::copy(var_58);
        // face_nrm = wp.cross(v1 - v0, v2 - v0)                                                  <L 664>
        var_59 = wp::sub(var_54, var_51);
        var_60 = wp::sub(var_57, var_51);
        var_61 = wp::cross(var_59, var_60);
        // face_nrm = wp.normalize(face_nrm)                                                      <L 665>
        var_62 = wp::normalize(var_61);
        // flexvert_norm_out[worldid, i0] += face_nrm                                             <L 666>
        var_63 = wp::atomic_add(var_flexvert_norm_out, var_0, var_38, var_62);
        // flexvert_norm_out[worldid, i1] += face_nrm                                             <L 667>
        var_64 = wp::atomic_add(var_flexvert_norm_out, var_0, var_43, var_62);
        // flexvert_norm_out[worldid, i2] += face_nrm                                             <L 668>
        var_65 = wp::atomic_add(var_flexvert_norm_out, var_0, var_48, var_62);
}

