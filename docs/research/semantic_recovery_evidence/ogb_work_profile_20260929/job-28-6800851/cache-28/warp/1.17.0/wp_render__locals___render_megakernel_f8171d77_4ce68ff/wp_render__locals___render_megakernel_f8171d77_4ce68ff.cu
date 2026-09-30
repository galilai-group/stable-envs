#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 64
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


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:32
static CUDA_CALLABLE void _ray_map_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def _ray_map(pos: wp.vec3, mat: wp.mat33, pnt: wp.vec3, vec: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:       <L 33>
    // matT = wp.transpose(mat)                                                               <L 45>
    var_0 = wp::transpose(var_mat);
    // lpnt = matT @ (pnt - pos)                                                              <L 46>
    var_1 = wp::sub(var_pnt, var_pos);
    var_2 = wp::mul(var_0, var_1);
    // lvec = matT @ vec                                                                      <L 47>
    var_3 = wp::mul(var_0, var_vec);
    // return lpnt, lvec                                                                      <L 49>
    ret_0 = var_2;
    ret_1 = var_3;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:213
static CUDA_CALLABLE void ray_plane_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-15;
    const wp::float32 var_5 = -1e-15;
    bool var_6;
    const wp::float32 var_7 = -1.0;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.0;
    bool var_16;
    const wp::float32 var_17 = -1.0;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::vec_t<2, wp::float32> var_31;
    bool var_32;
    bool var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::int32 var_38 = 0;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    bool var_43;
    bool var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    const wp::float32 var_47 = 0.0;
    bool var_48;
    const wp::int32 var_49 = 1;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 1;
    wp::float32 var_53;
    bool var_54;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 2;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::float32 var_65 = -1.0;
    wp::vec_t<3, wp::float32> var_66;
    //---------
    // forward
    // def ray_plane(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 214>
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 217>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_0, var_1);
    // if lvec[2] > -MJ_MINVAL:                                                               <L 220>
    var_3 = wp::extract(var_1, var_2);
    var_6 = (var_3 > var_5);
    if (var_6) {
        // return -1.0, wp.vec3()                                                             <L 221>
        var_8 = wp::vec_t<3, wp::float32>();
        ret_0 = var_7;
        ret_1 = var_8;
        return;
    }
    // x = -lpnt[2] / lvec[2]                                                                 <L 224>
    var_10 = wp::extract(var_0, var_9);
    var_11 = wp::neg(var_10);
    var_13 = wp::extract(var_1, var_12);
    var_14 = wp::div(var_11, var_13);
    // if x < 0.0:                                                                            <L 225>
    var_16 = (var_14 < var_15);
    if (var_16) {
        // return -1.0, wp.vec3()                                                             <L 226>
        var_18 = wp::vec_t<3, wp::float32>();
        ret_0 = var_17;
        ret_1 = var_18;
        return;
    }
    // p = wp.vec2(lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1])                              <L 228>
    var_20 = wp::extract(var_0, var_19);
    var_22 = wp::extract(var_1, var_21);
    var_23 = wp::mul(var_14, var_22);
    var_24 = wp::add(var_20, var_23);
    var_26 = wp::extract(var_0, var_25);
    var_28 = wp::extract(var_1, var_27);
    var_29 = wp::mul(var_14, var_28);
    var_30 = wp::add(var_26, var_29);
    var_31 = wp::vec_t<2, wp::float32>(var_24, var_30);
    // if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):       <L 231>
    var_35 = wp::extract(var_size, var_34);
    var_37 = (var_35 <= var_36);
    var_33 = var_37;
    if (!var_33) {
        var_39 = wp::extract(var_31, var_38);
        var_40 = wp::abs(var_39);
        var_42 = wp::extract(var_size, var_41);
        var_43 = (var_40 <= var_42);
        var_33 = var_33 || var_43;
    }
    var_32 = var_33;
    if (var_32) {
        var_46 = wp::extract(var_size, var_45);
        var_48 = (var_46 <= var_47);
        var_44 = var_48;
        if (!var_44) {
            var_50 = wp::extract(var_31, var_49);
            var_51 = wp::abs(var_50);
            var_53 = wp::extract(var_size, var_52);
            var_54 = (var_51 <= var_53);
            var_44 = var_44 || var_54;
        }
        var_32 = var_32 && var_44;
    }
    if (var_32) {
        // return x, wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])                                 <L 232>
        var_57 = wp::extract(var_mat, var_55, var_56);
        var_60 = wp::extract(var_mat, var_58, var_59);
        var_63 = wp::extract(var_mat, var_61, var_62);
        var_64 = wp::vec_t<3, wp::float32>(var_57, var_60, var_63);
        ret_0 = var_14;
        ret_1 = var_64;
        return;
    }
    if (!var_32) {
        // return -1.0, wp.vec3()                                                             <L 234>
        var_66 = wp::vec_t<3, wp::float32>();
        ret_0 = var_65;
        ret_1 = var_66;
        return;
    }
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:690
static CUDA_CALLABLE void ray_mesh_with_bvh_0(
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::int32 var_mesh_geom_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 var_max_t,
    bool var_cull_backfaces,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2,
    wp::float32 & ret_3,
    wp::int32 & ret_4,
    wp::int32 & ret_5)
{
    //---------
    // primal vars
    const wp::float32 var_0 = -1.0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.0;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = -1;
    wp::int32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::uint64* var_16;
    bool var_17;
    const wp::int32 var_18 = -1;
    wp::uint64 var_19;
    bool var_20;
    const wp::float32 var_21 = -1.0;
    const wp::float32 var_22 = 0.0;
    const wp::float32 var_23 = 0.0;
    const wp::float32 var_24 = 0.0;
    wp::vec_t<3, wp::float32> var_25;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    const wp::int32 var_28 = -1;
    const wp::int32 var_29 = -1;
    bool var_30;
    bool var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    bool var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    const wp::float32 var_37 = -1.0;
    const wp::float32 var_38 = 0.0;
    const wp::float32 var_39 = 0.0;
    const wp::float32 var_40 = 0.0;
    wp::vec_t<3, wp::float32> var_41;
    const wp::float32 var_42 = 0.0;
    const wp::float32 var_43 = 0.0;
    const wp::int32 var_44 = -1;
    const wp::int32 var_45 = -1;
    //---------
    // forward
    // def ray_mesh_with_bvh(                                                                 <L 691>
    // t = float(-1.0)                                                                        <L 709>
    var_1 = wp::float(var_0);
    // u = float(0.0)                                                                         <L 710>
    var_3 = wp::float(var_2);
    // v = float(0.0)                                                                         <L 711>
    var_5 = wp::float(var_4);
    // sign = float(0.0)                                                                      <L 712>
    var_7 = wp::float(var_6);
    // n = wp.vec3(0.0, 0.0, 0.0)                                                             <L 713>
    var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
    // f = int(-1)                                                                            <L 714>
    var_13 = wp::int(var_12);
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 716>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_14, var_15);
    // hit = wp.mesh_query_ray(mesh_bvh_id[mesh_geom_id], lpnt, lvec, max_t, t, u, v, sign, n, f)       <L 717>
    var_16 = wp::address(var_mesh_bvh_id, var_mesh_geom_id);
    var_19 = wp::load(var_16);
    var_17 = wp::mesh_query_ray(var_19, var_14, var_15, var_max_t, var_1, var_3, var_5, var_7, var_11, var_13, var_18);
    // if not hit:                                                                            <L 719>
    var_20 = wp::unot(var_17);
    if (var_20) {
        // return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1                              <L 720>
        var_25 = wp::vec_t<3, wp::float32>(var_22, var_23, var_24);
        ret_0 = var_21;
        ret_1 = var_25;
        ret_2 = var_26;
        ret_3 = var_27;
        ret_4 = var_28;
        ret_5 = var_29;
        return;
    }
    // if (not cull_backfaces) or wp.dot(lvec, n) < 0.0:                                      <L 724>
    var_31 = wp::unot(var_cull_backfaces);
    var_30 = var_31;
    if (!var_30) {
        var_32 = wp::dot(var_15, var_11);
        var_34 = (var_32 < var_33);
        var_30 = var_30 || var_34;
    }
    if (var_30) {
        // normal = mat @ n                                                                   <L 725>
        var_35 = wp::mul(var_mat, var_11);
        // normal = wp.normalize(normal)                                                      <L 726>
        var_36 = wp::normalize(var_35);
        // return t, normal, u, v, f, mesh_geom_id                                            <L 727>
        ret_0 = var_1;
        ret_1 = var_36;
        ret_2 = var_3;
        ret_3 = var_5;
        ret_4 = var_13;
        ret_5 = var_mesh_geom_id;
        return;
    }
    // return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1                                  <L 729>
    var_41 = wp::vec_t<3, wp::float32>(var_38, var_39, var_40);
    ret_0 = var_37;
    ret_1 = var_41;
    ret_2 = var_42;
    ret_3 = var_43;
    ret_4 = var_44;
    ret_5 = var_45;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:302
static CUDA_CALLABLE void _make_cast_ray__locals__cast_ray_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::uint64 var_bvh_id,
    wp::int32 var_group_root,
    wp::int32 var_worldid,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_flex_bvh_ngeom,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::array_t<wp::uint64> var_hfield_bvh_id,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::array_t<wp::uint64> var_flex_bvh_id,
    wp::array_t<wp::int32> var_flex_group_root,
    wp::vec_t<3, wp::float32> var_ray_origin_world,
    wp::vec_t<3, wp::float32> var_ray_dir_world,
    wp::float32 var_max_dist,
    bool var_cull_backfaces,
    wp::int32 & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & ret_3,
    wp::float32 & ret_4,
    wp::int32 & ret_5,
    wp::int32 & ret_6)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = -1;
    wp::int32 var_6;
    const wp::float32 var_7 = 0.0;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    wp::float32 var_10;
    const wp::int32 var_11 = -1;
    wp::int32 var_12;
    const wp::int32 var_13 = -1;
    wp::int32 var_14;
    wp::bvh_query_t var_15;
    const wp::int32 var_16 = 0;
    wp::int32 var_17;
    wp::int32 var_18;
    bool var_19;
    wp::int32 var_20;
    wp::int32 var_21;
    wp::int32 var_22;
    const wp::float32 var_23 = -1.0;
    wp::float32 var_24;
    const wp::int32 var_25 = -1;
    wp::int32 var_26;
    const wp::float32 var_27 = 0.0;
    wp::float32 var_28;
    const wp::float32 var_29 = 0.0;
    wp::float32 var_30;
    const wp::int32 var_31 = -1;
    wp::int32 var_32;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 0.0;
    wp::vec_t<3, wp::float32> var_36;
    const wp::int32 var_37 = -1;
    wp::int32 var_38;
    bool var_39;
    wp::int32* var_40;
    wp::int32 var_41;
    wp::int32 var_42;
    wp::int32* var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    const wp::int32 var_47 = 105;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    const bool var_51 = true;
    const wp::int32 var_52 = 0;
    bool var_53;
    wp::vec_t<3, wp::float32>* var_54;
    wp::mat_t<3, 3, wp::float32>* var_55;
    wp::shape_t* var_56;
    const wp::int32 var_57 = 0;
    wp::int32 var_58;
    wp::shape_t var_59;
    wp::int32 var_60;
    wp::vec_t<3, wp::float32>* var_61;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::mat_t<3, 3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::float32 var_67;
    wp::vec_t<3, wp::float32> var_68;
    const bool var_69 = false;
    const bool var_70 = false;
    const bool var_71 = false;
    const bool var_72 = false;
    const bool var_73 = false;
    const bool var_74 = false;
    const bool var_75 = true;
    const wp::int32 var_76 = 7;
    bool var_77;
    const bool var_78 = false;
    wp::shape_t* var_79;
    const wp::int32 var_80 = 0;
    wp::int32 var_81;
    wp::shape_t var_82;
    wp::int32 var_83;
    wp::int32* var_84;
    wp::vec_t<3, wp::float32>* var_85;
    wp::mat_t<3, 3, wp::float32>* var_86;
    wp::float32 var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::int32 var_91;
    wp::int32 var_92;
    wp::int32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::mat_t<3, 3, wp::float32> var_95;
    wp::float32 var_96;
    wp::int32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    wp::int32 var_100;
    wp::vec_t<3, wp::float32> var_101;
    const bool var_102 = false;
    bool var_103;
    const wp::float32 var_104 = 0.0;
    bool var_105;
    wp::float32 var_106;
    const wp::float32 var_107 = 0.0;
    bool var_108;
    const wp::float32 var_109 = -1.0;
    wp::float32 var_110;
    const bool var_111 = false;
    bool var_112;
    const wp::float32 var_113 = 0.0;
    bool var_114;
    bool var_115;
    wp::float32 var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::int32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::int32 var_121;
    wp::int32 var_122;
    wp::float32 var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::int32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::int32 var_128;
    wp::int32 var_129;
    //---------
    // forward
    // def cast_ray(                                                                          <L 303>
    // dist = max_dist                                                                        <L 333>
    var_0 = wp::copy(var_max_dist);
    // normal = wp.vec3(0.0, 0.0, 0.0)                                                        <L 334>
    var_4 = wp::vec_t<3, wp::float32>(var_1, var_2, var_3);
    // geom_id = int(-1)                                                                      <L 335>
    var_6 = wp::int(var_5);
    // bary_u = float(0.0)                                                                    <L 336>
    var_8 = wp::float(var_7);
    // bary_v = float(0.0)                                                                    <L 337>
    var_10 = wp::float(var_9);
    // face_idx = int(-1)                                                                     <L 338>
    var_12 = wp::int(var_11);
    // geom_mesh_id = int(-1)                                                                 <L 339>
    var_14 = wp::int(var_13);
    // query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)          <L 341>
    var_15 = wp::bvh_query_ray(var_bvh_id, var_ray_origin_world, var_ray_dir_world, var_group_root);
    // bounds_nr = int(0)                                                                     <L 342>
    var_17 = wp::int(var_16);
    // ngeom = bvh_ngeom + flex_bvh_ngeom                                                     <L 343>
    var_18 = wp::add(var_bvh_ngeom, var_flex_bvh_ngeom);
    // while wp.bvh_query_next(query, bounds_nr, dist):                                       <L 345>
    start_while_0:;
    var_19 = wp::bvh_query_ray_next(var_15, var_17, var_0);
    if ((var_19) == false) goto end_while_0;
        // gi_global = bounds_nr                                                              <L 346>
        var_20 = wp::copy(var_17);
        // local_id = gi_global - (worldid * ngeom)                                           <L 347>
        var_21 = wp::mul(var_worldid, var_18);
        var_22 = wp::sub(var_20, var_21);
        // d = float(-1.0)                                                                    <L 349>
        var_24 = wp::float(var_23);
        // hit_mesh_id = int(-1)                                                              <L 350>
        var_26 = wp::int(var_25);
        // u = float(0.0)                                                                     <L 351>
        var_28 = wp::float(var_27);
        // v = float(0.0)                                                                     <L 352>
        var_30 = wp::float(var_29);
        // f = int(-1)                                                                        <L 353>
        var_32 = wp::int(var_31);
        // n = wp.vec3(0.0, 0.0, 0.0)                                                         <L 354>
        var_36 = wp::vec_t<3, wp::float32>(var_33, var_34, var_35);
        // hit_geom_id = int(-1)                                                              <L 355>
        var_38 = wp::int(var_37);
        // if local_id < bvh_ngeom:                                                           <L 357>
        var_39 = (var_22 < var_bvh_ngeom);
        if (var_39) {
            // gi = enabled_geom_ids[local_id]                                                <L 358>
            var_40 = wp::address(var_enabled_geom_ids, var_22);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // gtype = geom_type[gi]                                                          <L 359>
            var_43 = wp::address(var_geom_type, var_41);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
        }
        if (!var_39) {
            // gi = local_id - bvh_ngeom                                                      <L 361>
            var_46 = wp::sub(var_22, var_bvh_ngeom);
            // gtype = GeomType.FLEX                                                          <L 362>
        }
        var_48 = wp::where(var_39, var_41, var_46);
        var_49 = wp::where(var_39, var_44, var_47);
        // hit_geom_id = gi                                                                   <L 364>
        var_50 = wp::copy(var_48);
        // if wp.static(int(GeomType.PLANE) in geom_ray_types):                               <L 366>
        // if gtype == GeomType.PLANE:                                                        <L 367>
        var_53 = (var_49 == var_52);
        if (var_53) {
            // d, n = ray_plane(                                                              <L 368>
            // geom_xpos_in[worldid, gi],                                                     <L 369>
            var_54 = wp::address(var_geom_xpos_in, var_worldid, var_48);
            // geom_xmat_in[worldid, gi],                                                     <L 370>
            var_55 = wp::address(var_geom_xmat_in, var_worldid, var_48);
            // geom_size[worldid % geom_size.shape[0], gi],                                   <L 371>
            var_56 = &(var_geom_size.shape);
            var_59 = wp::load(var_56);
            var_58 = wp::extract(var_59, var_57);
            var_60 = wp::mod(var_worldid, var_58);
            var_61 = wp::address(var_geom_size, var_60, var_48);
            // ray_origin_world,                                                              <L 372>
            // ray_dir_world,                                                                 <L 373>
            var_64 = wp::load(var_54);
            var_65 = wp::load(var_55);
            var_66 = wp::load(var_61);
            ray_plane_0(var_64, var_65, var_66, var_ray_origin_world, var_ray_dir_world, var_62, var_63);
        }
        var_67 = wp::where(var_53, var_62, var_24);
        var_68 = wp::where(var_53, var_63, var_36);
        // if wp.static(int(GeomType.HFIELD) in geom_ray_types):                              <L 375>
        // if wp.static(int(GeomType.SPHERE) in geom_ray_types):                              <L 387>
        // if wp.static(int(GeomType.ELLIPSOID) in geom_ray_types):                           <L 395>
        // if wp.static(int(GeomType.CAPSULE) in geom_ray_types):                             <L 404>
        // if wp.static(int(GeomType.CYLINDER) in geom_ray_types):                            <L 413>
        // if wp.static(int(GeomType.BOX) in geom_ray_types):                                 <L 422>
        // if wp.static(int(GeomType.MESH) in geom_ray_types):                                <L 431>
        // if gtype == GeomType.MESH:                                                         <L 432>
        var_77 = (var_49 == var_76);
        if (var_77) {
            // if wp.static(first_hit):                                                       <L 433>
            // d, n, u, v, f, hit_mesh_id = ray_mesh_with_bvh(                                <L 445>
            // mesh_bvh_id,                                                                   <L 446>
            // geom_dataid[worldid % geom_dataid.shape[0], gi],                               <L 447>
            var_79 = &(var_geom_dataid.shape);
            var_82 = wp::load(var_79);
            var_81 = wp::extract(var_82, var_80);
            var_83 = wp::mod(var_worldid, var_81);
            var_84 = wp::address(var_geom_dataid, var_83, var_48);
            // geom_xpos_in[worldid, gi],                                                     <L 448>
            var_85 = wp::address(var_geom_xpos_in, var_worldid, var_48);
            // geom_xmat_in[worldid, gi],                                                     <L 449>
            var_86 = wp::address(var_geom_xmat_in, var_worldid, var_48);
            // ray_origin_world,                                                              <L 450>
            // ray_dir_world,                                                                 <L 451>
            // dist,                                                                          <L 452>
            // cull_backfaces,                                                                <L 453>
            var_93 = wp::load(var_84);
            var_94 = wp::load(var_85);
            var_95 = wp::load(var_86);
            ray_mesh_with_bvh_0(var_mesh_bvh_id, var_93, var_94, var_95, var_ray_origin_world, var_ray_dir_world, var_0, var_cull_backfaces, var_87, var_88, var_89, var_90, var_91, var_92);
        }
        var_96 = wp::where(var_77, var_87, var_67);
        var_97 = wp::where(var_77, var_92, var_26);
        var_98 = wp::where(var_77, var_89, var_28);
        var_99 = wp::where(var_77, var_90, var_30);
        var_100 = wp::where(var_77, var_91, var_32);
        var_101 = wp::where(var_77, var_88, var_68);
        // if wp.static(int(GeomType.FLEX) in geom_ray_types):                                <L 455>
        // if cull_backfaces and d >= 0.0 and wp.dot(ray_dir_world, n) > 0.0:                 <L 496>
        var_103 = var_cull_backfaces;
        if (var_103) {
            var_105 = (var_96 >= var_104);
            var_103 = var_103 && var_105;
        }
        if (var_103) {
            var_106 = wp::dot(var_ray_dir_world, var_101);
            var_108 = (var_106 > var_107);
            var_103 = var_103 && var_108;
        }
        if (var_103) {
            // d = -1.0                                                                       <L 497>
        }
        var_110 = wp::where(var_103, var_109, var_96);
        // if wp.static(first_hit):                                                           <L 499>
        // if d >= 0.0 and d < dist:                                                          <L 504>
        var_114 = (var_110 >= var_113);
        var_112 = var_114;
        if (var_112) {
            var_115 = (var_110 < var_0);
            var_112 = var_112 && var_115;
        }
        if (var_112) {
            // dist = d                                                                       <L 505>
            var_116 = wp::copy(var_110);
            // normal = n                                                                     <L 506>
            var_117 = wp::copy(var_101);
            // geom_id = hit_geom_id                                                          <L 507>
            var_118 = wp::copy(var_50);
            // bary_u = u                                                                     <L 508>
            var_119 = wp::copy(var_98);
            // bary_v = v                                                                     <L 509>
            var_120 = wp::copy(var_99);
            // face_idx = f                                                                   <L 510>
            var_121 = wp::copy(var_100);
            // geom_mesh_id = hit_mesh_id                                                     <L 511>
            var_122 = wp::copy(var_97);
        }
        var_123 = wp::where(var_112, var_116, var_0);
        var_124 = wp::where(var_112, var_117, var_4);
        var_125 = wp::where(var_112, var_118, var_6);
        var_126 = wp::where(var_112, var_119, var_8);
        var_127 = wp::where(var_112, var_120, var_10);
        var_128 = wp::where(var_112, var_121, var_12);
        var_129 = wp::where(var_112, var_122, var_14);
        wp::assign(var_0, var_123);
        wp::assign(var_4, var_124);
        wp::assign(var_6, var_125);
        wp::assign(var_8, var_126);
        wp::assign(var_10, var_127);
        wp::assign(var_12, var_128);
        wp::assign(var_14, var_129);
    goto start_while_0;
    end_while_0:;
    // return geom_id, dist, normal, bary_u, bary_v, face_idx, geom_mesh_id                   <L 513>
    ret_0 = var_6;
    ret_1 = var_0;
    ret_2 = var_4;
    ret_3 = var_8;
    ret_4 = var_10;
    ret_5 = var_12;
    ret_6 = var_14;
    return;
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


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:222
static CUDA_CALLABLE wp::vec_t<3, wp::float32> sample_skybox_0(
    wp::texture2d_t var_skybox_tex,
    wp::float32 var_face_width_inv,
    wp::vec_t<3, wp::float32> var_ray_dir_world)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::int32 var_11;
    const wp::float32 var_12 = 0.0;
    wp::float32 var_13;
    const wp::float32 var_14 = 0.0;
    wp::float32 var_15;
    const wp::float32 var_16 = 1.0;
    wp::float32 var_17;
    bool var_18;
    bool var_19;
    bool var_20;
    wp::float32 var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    bool var_33;
    wp::float32 var_34;
    const wp::float32 var_35 = 0.0;
    bool var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::int32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 0.0;
    bool var_48;
    const wp::int32 var_49 = 4;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 5;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::int32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::int32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::int32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::float32 var_67 = 1.0;
    wp::float32 var_68;
    const wp::float32 var_69 = 0.5;
    wp::float32 var_70;
    wp::float32 var_71;
    const wp::float32 var_72 = 1.0;
    wp::float32 var_73;
    const wp::float32 var_74 = 0.5;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.5;
    wp::float32 var_77;
    const wp::float32 var_78 = 1.0;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    const wp::float32 var_83 = 0.16666666666666666;
    wp::float32 var_84;
    wp::vec_t<2, wp::float32> var_85;
    wp::vec_t<4, wp::float32> var_86;
    const wp::float32 var_87 = -1.0;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    const wp::int32 var_92 = 2;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    //---------
    // forward
    // def sample_skybox(                                                                     <L 223>
    // rx = ray_dir_world[0]                                                                  <L 232>
    var_1 = wp::extract(var_ray_dir_world, var_0);
    // ry = ray_dir_world[2]                                                                  <L 233>
    var_3 = wp::extract(var_ray_dir_world, var_2);
    // rz = -ray_dir_world[1]                                                                 <L 234>
    var_5 = wp::extract(var_ray_dir_world, var_4);
    var_6 = wp::neg(var_5);
    // arx = wp.abs(rx)                                                                       <L 236>
    var_7 = wp::abs(var_1);
    // ary = wp.abs(ry)                                                                       <L 237>
    var_8 = wp::abs(var_3);
    // arz = wp.abs(rz)                                                                       <L 238>
    var_9 = wp::abs(var_6);
    // face = int(0)                                                                          <L 240>
    var_11 = wp::int(var_10);
    // sc = float(0.0)                                                                        <L 241>
    var_13 = wp::float(var_12);
    // tc = float(0.0)                                                                        <L 242>
    var_15 = wp::float(var_14);
    // ma = float(1.0)                                                                        <L 243>
    var_17 = wp::float(var_16);
    // if arx >= ary and arx >= arz:                                                          <L 245>
    var_19 = (var_7 >= var_8);
    var_18 = var_19;
    if (var_18) {
        var_20 = (var_7 >= var_9);
        var_18 = var_18 && var_20;
    }
    if (var_18) {
        // ma = arx                                                                           <L 246>
        var_21 = wp::copy(var_7);
        // if rx > 0.0:                                                                       <L 247>
        var_23 = (var_1 > var_22);
        if (var_23) {
            // face = 0                                                                       <L 248>
            // sc = -rz                                                                       <L 249>
            var_25 = wp::neg(var_6);
            // tc = -ry                                                                       <L 250>
            var_26 = wp::neg(var_3);
        }
        if (!var_23) {
            // face = 1                                                                       <L 252>
            // sc = rz                                                                        <L 253>
            var_28 = wp::copy(var_6);
            // tc = -ry                                                                       <L 254>
            var_29 = wp::neg(var_3);
        }
        var_30 = wp::where(var_23, var_24, var_27);
        var_31 = wp::where(var_23, var_25, var_28);
        var_32 = wp::where(var_23, var_26, var_29);
    }
    if (!var_18) {
        // elif ary >= arz:                                                                   <L 255>
        var_33 = (var_8 >= var_9);
        if (var_33) {
            // ma = ary                                                                       <L 256>
            var_34 = wp::copy(var_8);
            // if ry > 0.0:                                                                   <L 257>
            var_36 = (var_3 > var_35);
            if (var_36) {
                // face = 2                                                                   <L 258>
                // sc = rx                                                                    <L 259>
                var_38 = wp::copy(var_1);
                // tc = rz                                                                    <L 260>
                var_39 = wp::copy(var_6);
            }
            if (!var_36) {
                // face = 3                                                                   <L 262>
                // sc = rx                                                                    <L 263>
                var_41 = wp::copy(var_1);
                // tc = -rz                                                                   <L 264>
                var_42 = wp::neg(var_6);
            }
            var_43 = wp::where(var_36, var_37, var_40);
            var_44 = wp::where(var_36, var_38, var_41);
            var_45 = wp::where(var_36, var_39, var_42);
        }
        if (!var_33) {
            // ma = arz                                                                       <L 266>
            var_46 = wp::copy(var_9);
            // if rz > 0.0:                                                                   <L 267>
            var_48 = (var_6 > var_47);
            if (var_48) {
                // face = 4                                                                   <L 268>
                // sc = rx                                                                    <L 269>
                var_50 = wp::copy(var_1);
                // tc = -ry                                                                   <L 270>
                var_51 = wp::neg(var_3);
            }
            if (!var_48) {
                // face = 5                                                                   <L 272>
                // sc = -rx                                                                   <L 273>
                var_53 = wp::neg(var_1);
                // tc = -ry                                                                   <L 274>
                var_54 = wp::neg(var_3);
            }
            var_55 = wp::where(var_48, var_49, var_52);
            var_56 = wp::where(var_48, var_50, var_53);
            var_57 = wp::where(var_48, var_51, var_54);
        }
        var_58 = wp::where(var_33, var_43, var_55);
        var_59 = wp::where(var_33, var_44, var_56);
        var_60 = wp::where(var_33, var_45, var_57);
        var_61 = wp::where(var_33, var_34, var_46);
    }
    var_62 = wp::where(var_18, var_30, var_58);
    var_63 = wp::where(var_18, var_31, var_59);
    var_64 = wp::where(var_18, var_32, var_60);
    var_65 = wp::where(var_18, var_21, var_61);
    // s = (math.safe_div(sc, ma) + 1.0) * 0.5                                                <L 276>
    var_66 = safe_div_0(var_63, var_65);
    var_68 = wp::add(var_66, var_67);
    var_70 = wp::mul(var_68, var_69);
    // t = (math.safe_div(tc, ma) + 1.0) * 0.5                                                <L 277>
    var_71 = safe_div_0(var_64, var_65);
    var_73 = wp::add(var_71, var_72);
    var_75 = wp::mul(var_73, var_74);
    // t_min = 0.5 * face_width_inv                                                           <L 280>
    var_77 = wp::mul(var_76, var_face_width_inv);
    // t = wp.clamp(t, t_min, 1.0 - t_min)                                                    <L 281>
    var_79 = wp::sub(var_78, var_77);
    var_80 = wp::clamp(var_75, var_77, var_79);
    // v = (float(face) + t) * wp.static(1.0 / 6.0)                                           <L 283>
    var_81 = wp::float(var_62);
    var_82 = wp::add(var_81, var_80);
    var_84 = wp::mul(var_82, var_83);
    // color = wp.texture_sample(skybox_tex, wp.vec2(s, v), dtype=wp.vec4)                    <L 284>
    var_85 = wp::vec_t<2, wp::float32>(var_70, var_84);
    var_86 = wp::texture_sample<wp::vec_t<4, wp::float32>>(var_skybox_tex, var_85, var_87);
    // return wp.vec3(color[0], color[1], color[2])                                           <L 285>
    var_89 = wp::extract(var_86, var_88);
    var_91 = wp::extract(var_86, var_90);
    var_93 = wp::extract(var_86, var_92);
    var_94 = wp::vec_t<3, wp::float32>(var_89, var_91, var_93);
    return var_94;
}


// /oscar/scratch/sboughan/se-semantic-recovery/render-parity-df80c9c5-20260928/source-v12/stable_envs/render/mjwarp.py:755
static CUDA_CALLABLE wp::uint32 gl_rounding__locals__pack_rgba_round_0(
    wp::float32 var_r,
    wp::float32 var_g,
    wp::float32 var_b,
    wp::float32 var_a)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 24;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::float32 var_4 = 0.5;
    wp::float32 var_5;
    wp::int32 var_6;
    const wp::int32 var_7 = 16;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    const wp::float32 var_11 = 0.5;
    wp::float32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 8;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    const wp::float32 var_18 = 0.5;
    wp::float32 var_19;
    wp::int32 var_20;
    wp::int32 var_21;
    wp::uint32 var_22;
    //---------
    // forward
    // def pack_rgba_round(r: float, g: float, b: float, a: float) -> wp.uint32:              <L 756>
    // return wp.uint32((int(a) << int(24)) | (int(r + 0.5) << int(16)) | (int(g + 0.5) << int(8)) | int(b + 0.5))       <L 757>
    var_0 = wp::int(var_a);
    var_2 = wp::int(var_1);
    var_3 = wp::lshift(var_0, var_2);
    var_5 = wp::add(var_r, var_4);
    var_6 = wp::int(var_5);
    var_8 = wp::int(var_7);
    var_9 = wp::lshift(var_6, var_8);
    var_10 = wp::bit_or(var_3, var_9);
    var_12 = wp::add(var_g, var_11);
    var_13 = wp::int(var_12);
    var_15 = wp::int(var_14);
    var_16 = wp::lshift(var_13, var_15);
    var_17 = wp::bit_or(var_10, var_16);
    var_19 = wp::add(var_b, var_18);
    var_20 = wp::int(var_19);
    var_21 = wp::bit_or(var_17, var_20);
    var_22 = wp::uint32(var_21);
    return var_22;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:172
static CUDA_CALLABLE wp::vec_t<3, wp::float32> sample_texture_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_mesh_faceadr,
    wp::int32 var_geom_id,
    wp::vec_t<2, wp::float32> var_tex_repeat,
    wp::texture2d_t var_tex,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::array_t<wp::vec_t<3, wp::int32>> var_mesh_facetexcoord,
    wp::array_t<wp::vec_t<2, wp::float32>> var_mesh_texcoord,
    wp::array_t<wp::int32> var_mesh_texcoord_offsets,
    wp::vec_t<3, wp::float32> var_hit_point,
    wp::float32 var_bary_u,
    wp::float32 var_bary_v,
    wp::int32 var_f,
    wp::int32 var_mesh_id)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    wp::vec_t<2, wp::float32> var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<2, wp::float32> var_5;
    wp::int32* var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    wp::int32 var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 0.5;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = -0.5;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<2, wp::float32> var_21;
    const wp::float32 var_22 = -0.5;
    const wp::float32 var_23 = -0.5;
    wp::vec_t<2, wp::float32> var_24;
    wp::vec_t<2, wp::float32> var_25;
    wp::vec_t<2, wp::float32> var_26;
    wp::int32* var_27;
    const wp::int32 var_28 = 7;
    bool var_29;
    wp::int32 var_30;
    bool var_31;
    const wp::int32 var_32 = 0;
    bool var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    const wp::float32 var_36 = 0.0;
    const wp::float32 var_37 = 0.0;
    const wp::float32 var_38 = 0.0;
    wp::vec_t<3, wp::float32> var_39;
    wp::int32* var_40;
    wp::int32 var_41;
    wp::int32 var_42;
    wp::int32* var_43;
    wp::vec_t<3, wp::int32>* var_44;
    const wp::int32 var_45 = 0;
    wp::int32 var_46;
    wp::vec_t<3, wp::int32> var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::vec_t<2, wp::float32>* var_50;
    wp::vec_t<2, wp::float32> var_51;
    wp::vec_t<2, wp::float32> var_52;
    wp::int32* var_53;
    wp::vec_t<3, wp::int32>* var_54;
    const wp::int32 var_55 = 1;
    wp::int32 var_56;
    wp::vec_t<3, wp::int32> var_57;
    wp::int32 var_58;
    wp::int32 var_59;
    wp::vec_t<2, wp::float32>* var_60;
    wp::vec_t<2, wp::float32> var_61;
    wp::vec_t<2, wp::float32> var_62;
    wp::int32* var_63;
    wp::vec_t<3, wp::int32>* var_64;
    const wp::int32 var_65 = 2;
    wp::int32 var_66;
    wp::vec_t<3, wp::int32> var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    wp::vec_t<2, wp::float32>* var_70;
    wp::vec_t<2, wp::float32> var_71;
    wp::vec_t<2, wp::float32> var_72;
    wp::vec_t<2, wp::float32> var_73;
    wp::vec_t<2, wp::float32> var_74;
    wp::vec_t<2, wp::float32> var_75;
    const wp::float32 var_76 = 1.0;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::vec_t<2, wp::float32> var_79;
    wp::vec_t<2, wp::float32> var_80;
    wp::vec_t<2, wp::float32> var_81;
    const wp::int32 var_82 = 0;
    wp::float32 var_83;
    const wp::int32 var_84 = 0;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 0;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    wp::float32 var_94;
    const wp::int32 var_95 = 1;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::vec_t<2, wp::float32> var_102;
    wp::vec_t<4, wp::float32> var_103;
    const wp::float32 var_104 = -1.0;
    const wp::int32 var_105 = 0;
    wp::float32 var_106;
    const wp::int32 var_107 = 1;
    wp::float32 var_108;
    const wp::int32 var_109 = 2;
    wp::float32 var_110;
    wp::vec_t<3, wp::float32> var_111;
    //---------
    // forward
    // def sample_texture(                                                                    <L 173>
    // uv = wp.vec2(0.0, 0.0)                                                                 <L 192>
    var_2 = wp::vec_t<2, wp::float32>(var_0, var_1);
    // offset = wp.vec2(0.0, 0.0)                                                             <L 193>
    var_5 = wp::vec_t<2, wp::float32>(var_3, var_4);
    // if geom_type[geom_id] == GeomType.PLANE:                                               <L 195>
    var_6 = wp::address(var_geom_type, var_geom_id);
    var_9 = wp::load(var_6);
    var_8 = (var_9 == var_7);
    if (var_8) {
        // local = wp.transpose(rot) @ (hit_point - pos)                                      <L 196>
        var_10 = wp::transpose(var_rot);
        var_11 = wp::sub(var_hit_point, var_pos);
        var_12 = wp::mul(var_10, var_11);
        // uv = wp.vec2(0.5 * local[0], -0.5 * local[1])                                      <L 201>
        var_15 = wp::extract(var_12, var_14);
        var_16 = wp::mul(var_13, var_15);
        var_19 = wp::extract(var_12, var_18);
        var_20 = wp::mul(var_17, var_19);
        var_21 = wp::vec_t<2, wp::float32>(var_16, var_20);
        // offset = wp.vec2(-0.5, -0.5)                                                       <L 202>
        var_24 = wp::vec_t<2, wp::float32>(var_22, var_23);
    }
    var_25 = wp::where(var_8, var_21, var_2);
    var_26 = wp::where(var_8, var_24, var_5);
    // if geom_type[geom_id] == GeomType.MESH:                                                <L 204>
    var_27 = wp::address(var_geom_type, var_geom_id);
    var_30 = wp::load(var_27);
    var_29 = (var_30 == var_28);
    if (var_29) {
        // if f < 0 or mesh_id < 0:                                                           <L 205>
        var_33 = (var_f < var_32);
        var_31 = var_33;
        if (!var_31) {
            var_35 = (var_mesh_id < var_34);
            var_31 = var_31 || var_35;
        }
        if (var_31) {
            // return wp.vec3(0.0, 0.0, 0.0)                                                  <L 206>
            var_39 = wp::vec_t<3, wp::float32>(var_36, var_37, var_38);
            return var_39;
        }
        // face_adr = mesh_faceadr[mesh_id] + f                                               <L 208>
        var_40 = wp::address(var_mesh_faceadr, var_mesh_id);
        var_42 = wp::load(var_40);
        var_41 = wp::add(var_42, var_f);
        // uv0 = mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][0]]       <L 209>
        var_43 = wp::address(var_mesh_texcoord_offsets, var_mesh_id);
        var_44 = wp::address(var_mesh_facetexcoord, var_41);
        var_47 = wp::load(var_44);
        var_46 = wp::extract(var_47, var_45);
        var_49 = wp::load(var_43);
        var_48 = wp::add(var_49, var_46);
        var_50 = wp::address(var_mesh_texcoord, var_48);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // uv1 = mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][1]]       <L 210>
        var_53 = wp::address(var_mesh_texcoord_offsets, var_mesh_id);
        var_54 = wp::address(var_mesh_facetexcoord, var_41);
        var_57 = wp::load(var_54);
        var_56 = wp::extract(var_57, var_55);
        var_59 = wp::load(var_53);
        var_58 = wp::add(var_59, var_56);
        var_60 = wp::address(var_mesh_texcoord, var_58);
        var_62 = wp::load(var_60);
        var_61 = wp::copy(var_62);
        // uv2 = mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][2]]       <L 211>
        var_63 = wp::address(var_mesh_texcoord_offsets, var_mesh_id);
        var_64 = wp::address(var_mesh_facetexcoord, var_41);
        var_67 = wp::load(var_64);
        var_66 = wp::extract(var_67, var_65);
        var_69 = wp::load(var_63);
        var_68 = wp::add(var_69, var_66);
        var_70 = wp::address(var_mesh_texcoord, var_68);
        var_72 = wp::load(var_70);
        var_71 = wp::copy(var_72);
        // uv = uv0 * bary_u + uv1 * bary_v + uv2 * (1.0 - bary_u - bary_v)                   <L 212>
        var_73 = wp::mul(var_51, var_bary_u);
        var_74 = wp::mul(var_61, var_bary_v);
        var_75 = wp::add(var_73, var_74);
        var_77 = wp::sub(var_76, var_bary_u);
        var_78 = wp::sub(var_77, var_bary_v);
        var_79 = wp::mul(var_71, var_78);
        var_80 = wp::add(var_75, var_79);
    }
    var_81 = wp::where(var_29, var_80, var_25);
    // u = uv[0] * tex_repeat[0] + offset[0]                                                  <L 214>
    var_83 = wp::extract(var_81, var_82);
    var_85 = wp::extract(var_tex_repeat, var_84);
    var_86 = wp::mul(var_83, var_85);
    var_88 = wp::extract(var_26, var_87);
    var_89 = wp::add(var_86, var_88);
    // v = uv[1] * tex_repeat[1] + offset[1]                                                  <L 215>
    var_91 = wp::extract(var_81, var_90);
    var_93 = wp::extract(var_tex_repeat, var_92);
    var_94 = wp::mul(var_91, var_93);
    var_96 = wp::extract(var_26, var_95);
    var_97 = wp::add(var_94, var_96);
    // u = u - wp.floor(u)                                                                    <L 216>
    var_98 = wp::floor(var_89);
    var_99 = wp::sub(var_89, var_98);
    // v = v - wp.floor(v)                                                                    <L 217>
    var_100 = wp::floor(var_97);
    var_101 = wp::sub(var_97, var_100);
    // tex_color = wp.texture_sample(tex, wp.vec2(u, v), dtype=wp.vec4)                       <L 218>
    var_102 = wp::vec_t<2, wp::float32>(var_99, var_101);
    var_103 = wp::texture_sample<wp::vec_t<4, wp::float32>>(var_tex, var_102, var_104);
    // return wp.vec3(tex_color[0], tex_color[1], tex_color[2])                               <L 219>
    var_106 = wp::extract(var_103, var_105);
    var_108 = wp::extract(var_103, var_107);
    var_110 = wp::extract(var_103, var_109);
    var_111 = wp::vec_t<3, wp::float32>(var_106, var_108, var_110);
    return var_111;
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


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:732
static CUDA_CALLABLE bool ray_mesh_with_bvh_anyhit_0(
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::int32 var_mesh_geom_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 var_max_t)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::uint64* var_2;
    bool var_3;
    const wp::int32 var_4 = -1;
    wp::uint64 var_5;
    //---------
    // forward
    // def ray_mesh_with_bvh_anyhit(                                                          <L 733>
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 748>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_0, var_1);
    // return wp.mesh_query_ray_anyhit(mesh_bvh_id[mesh_geom_id], lpnt, lvec, max_t)          <L 749>
    var_2 = wp::address(var_mesh_bvh_id, var_mesh_geom_id);
    var_5 = wp::load(var_2);
    var_3 = wp::mesh_query_ray_anyhit(var_5, var_0, var_1, var_max_t, var_4);
    return var_3;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:302
static CUDA_CALLABLE void _make_cast_ray__locals__cast_ray_1(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::uint64 var_bvh_id,
    wp::int32 var_group_root,
    wp::int32 var_worldid,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_flex_bvh_ngeom,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::array_t<wp::uint64> var_hfield_bvh_id,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::array_t<wp::uint64> var_flex_bvh_id,
    wp::array_t<wp::int32> var_flex_group_root,
    wp::vec_t<3, wp::float32> var_ray_origin_world,
    wp::vec_t<3, wp::float32> var_ray_dir_world,
    wp::float32 var_max_dist,
    bool var_cull_backfaces,
    wp::int32 & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & ret_3,
    wp::float32 & ret_4,
    wp::int32 & ret_5,
    wp::int32 & ret_6)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = -1;
    wp::int32 var_6;
    const wp::float32 var_7 = 0.0;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    wp::float32 var_10;
    const wp::int32 var_11 = -1;
    wp::int32 var_12;
    const wp::int32 var_13 = -1;
    wp::int32 var_14;
    wp::bvh_query_t var_15;
    const wp::int32 var_16 = 0;
    wp::int32 var_17;
    wp::int32 var_18;
    bool var_19;
    wp::int32 var_20;
    wp::int32 var_21;
    wp::int32 var_22;
    const wp::float32 var_23 = -1.0;
    wp::float32 var_24;
    const wp::int32 var_25 = -1;
    wp::int32 var_26;
    const wp::float32 var_27 = 0.0;
    wp::float32 var_28;
    const wp::float32 var_29 = 0.0;
    wp::float32 var_30;
    const wp::int32 var_31 = -1;
    wp::int32 var_32;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 0.0;
    wp::vec_t<3, wp::float32> var_36;
    const wp::int32 var_37 = -1;
    wp::int32 var_38;
    bool var_39;
    wp::int32* var_40;
    wp::int32 var_41;
    wp::int32 var_42;
    wp::int32* var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    const wp::int32 var_47 = 105;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    const bool var_51 = true;
    const wp::int32 var_52 = 0;
    bool var_53;
    wp::vec_t<3, wp::float32>* var_54;
    wp::mat_t<3, 3, wp::float32>* var_55;
    wp::shape_t* var_56;
    const wp::int32 var_57 = 0;
    wp::int32 var_58;
    wp::shape_t var_59;
    wp::int32 var_60;
    wp::vec_t<3, wp::float32>* var_61;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::mat_t<3, 3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::float32 var_67;
    wp::vec_t<3, wp::float32> var_68;
    const bool var_69 = false;
    const bool var_70 = false;
    const bool var_71 = false;
    const bool var_72 = false;
    const bool var_73 = false;
    const bool var_74 = false;
    const bool var_75 = true;
    const wp::int32 var_76 = 7;
    bool var_77;
    const bool var_78 = true;
    wp::shape_t* var_79;
    const wp::int32 var_80 = 0;
    wp::int32 var_81;
    wp::shape_t var_82;
    wp::int32 var_83;
    wp::int32* var_84;
    wp::vec_t<3, wp::float32>* var_85;
    wp::mat_t<3, 3, wp::float32>* var_86;
    bool var_87;
    wp::int32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::mat_t<3, 3, wp::float32> var_90;
    const wp::float32 var_91 = 0.0;
    const wp::float32 var_92 = -1.0;
    wp::float32 var_93;
    wp::float32 var_94;
    const bool var_95 = false;
    bool var_96;
    const wp::float32 var_97 = 0.0;
    bool var_98;
    wp::float32 var_99;
    const wp::float32 var_100 = 0.0;
    bool var_101;
    const wp::float32 var_102 = -1.0;
    wp::float32 var_103;
    const bool var_104 = true;
    bool var_105;
    const wp::float32 var_106 = 0.0;
    bool var_107;
    bool var_108;
    //---------
    // forward
    // def cast_ray(                                                                          <L 303>
    // dist = max_dist                                                                        <L 333>
    var_0 = wp::copy(var_max_dist);
    // normal = wp.vec3(0.0, 0.0, 0.0)                                                        <L 334>
    var_4 = wp::vec_t<3, wp::float32>(var_1, var_2, var_3);
    // geom_id = int(-1)                                                                      <L 335>
    var_6 = wp::int(var_5);
    // bary_u = float(0.0)                                                                    <L 336>
    var_8 = wp::float(var_7);
    // bary_v = float(0.0)                                                                    <L 337>
    var_10 = wp::float(var_9);
    // face_idx = int(-1)                                                                     <L 338>
    var_12 = wp::int(var_11);
    // geom_mesh_id = int(-1)                                                                 <L 339>
    var_14 = wp::int(var_13);
    // query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)          <L 341>
    var_15 = wp::bvh_query_ray(var_bvh_id, var_ray_origin_world, var_ray_dir_world, var_group_root);
    // bounds_nr = int(0)                                                                     <L 342>
    var_17 = wp::int(var_16);
    // ngeom = bvh_ngeom + flex_bvh_ngeom                                                     <L 343>
    var_18 = wp::add(var_bvh_ngeom, var_flex_bvh_ngeom);
    // while wp.bvh_query_next(query, bounds_nr, dist):                                       <L 345>
    start_while_0:;
    var_19 = wp::bvh_query_ray_next(var_15, var_17, var_0);
    if ((var_19) == false) goto end_while_0;
        // gi_global = bounds_nr                                                              <L 346>
        var_20 = wp::copy(var_17);
        // local_id = gi_global - (worldid * ngeom)                                           <L 347>
        var_21 = wp::mul(var_worldid, var_18);
        var_22 = wp::sub(var_20, var_21);
        // d = float(-1.0)                                                                    <L 349>
        var_24 = wp::float(var_23);
        // hit_mesh_id = int(-1)                                                              <L 350>
        var_26 = wp::int(var_25);
        // u = float(0.0)                                                                     <L 351>
        var_28 = wp::float(var_27);
        // v = float(0.0)                                                                     <L 352>
        var_30 = wp::float(var_29);
        // f = int(-1)                                                                        <L 353>
        var_32 = wp::int(var_31);
        // n = wp.vec3(0.0, 0.0, 0.0)                                                         <L 354>
        var_36 = wp::vec_t<3, wp::float32>(var_33, var_34, var_35);
        // hit_geom_id = int(-1)                                                              <L 355>
        var_38 = wp::int(var_37);
        // if local_id < bvh_ngeom:                                                           <L 357>
        var_39 = (var_22 < var_bvh_ngeom);
        if (var_39) {
            // gi = enabled_geom_ids[local_id]                                                <L 358>
            var_40 = wp::address(var_enabled_geom_ids, var_22);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // gtype = geom_type[gi]                                                          <L 359>
            var_43 = wp::address(var_geom_type, var_41);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
        }
        if (!var_39) {
            // gi = local_id - bvh_ngeom                                                      <L 361>
            var_46 = wp::sub(var_22, var_bvh_ngeom);
            // gtype = GeomType.FLEX                                                          <L 362>
        }
        var_48 = wp::where(var_39, var_41, var_46);
        var_49 = wp::where(var_39, var_44, var_47);
        // hit_geom_id = gi                                                                   <L 364>
        var_50 = wp::copy(var_48);
        // if wp.static(int(GeomType.PLANE) in geom_ray_types):                               <L 366>
        // if gtype == GeomType.PLANE:                                                        <L 367>
        var_53 = (var_49 == var_52);
        if (var_53) {
            // d, n = ray_plane(                                                              <L 368>
            // geom_xpos_in[worldid, gi],                                                     <L 369>
            var_54 = wp::address(var_geom_xpos_in, var_worldid, var_48);
            // geom_xmat_in[worldid, gi],                                                     <L 370>
            var_55 = wp::address(var_geom_xmat_in, var_worldid, var_48);
            // geom_size[worldid % geom_size.shape[0], gi],                                   <L 371>
            var_56 = &(var_geom_size.shape);
            var_59 = wp::load(var_56);
            var_58 = wp::extract(var_59, var_57);
            var_60 = wp::mod(var_worldid, var_58);
            var_61 = wp::address(var_geom_size, var_60, var_48);
            // ray_origin_world,                                                              <L 372>
            // ray_dir_world,                                                                 <L 373>
            var_64 = wp::load(var_54);
            var_65 = wp::load(var_55);
            var_66 = wp::load(var_61);
            ray_plane_0(var_64, var_65, var_66, var_ray_origin_world, var_ray_dir_world, var_62, var_63);
        }
        var_67 = wp::where(var_53, var_62, var_24);
        var_68 = wp::where(var_53, var_63, var_36);
        // if wp.static(int(GeomType.HFIELD) in geom_ray_types):                              <L 375>
        // if wp.static(int(GeomType.SPHERE) in geom_ray_types):                              <L 387>
        // if wp.static(int(GeomType.ELLIPSOID) in geom_ray_types):                           <L 395>
        // if wp.static(int(GeomType.CAPSULE) in geom_ray_types):                             <L 404>
        // if wp.static(int(GeomType.CYLINDER) in geom_ray_types):                            <L 413>
        // if wp.static(int(GeomType.BOX) in geom_ray_types):                                 <L 422>
        // if wp.static(int(GeomType.MESH) in geom_ray_types):                                <L 431>
        // if gtype == GeomType.MESH:                                                         <L 432>
        var_77 = (var_49 == var_76);
        if (var_77) {
            // if wp.static(first_hit):                                                       <L 433>
            // hit = ray_mesh_with_bvh_anyhit(                                                <L 434>
            // mesh_bvh_id,                                                                   <L 435>
            // geom_dataid[worldid % geom_dataid.shape[0], gi],                               <L 436>
            var_79 = &(var_geom_dataid.shape);
            var_82 = wp::load(var_79);
            var_81 = wp::extract(var_82, var_80);
            var_83 = wp::mod(var_worldid, var_81);
            var_84 = wp::address(var_geom_dataid, var_83, var_48);
            // geom_xpos_in[worldid, gi],                                                     <L 437>
            var_85 = wp::address(var_geom_xpos_in, var_worldid, var_48);
            // geom_xmat_in[worldid, gi],                                                     <L 438>
            var_86 = wp::address(var_geom_xmat_in, var_worldid, var_48);
            // ray_origin_world,                                                              <L 439>
            // ray_dir_world,                                                                 <L 440>
            // dist,                                                                          <L 441>
            var_88 = wp::load(var_84);
            var_89 = wp::load(var_85);
            var_90 = wp::load(var_86);
            var_87 = ray_mesh_with_bvh_anyhit_0(var_mesh_bvh_id, var_88, var_89, var_90, var_ray_origin_world, var_ray_dir_world, var_0);
            // d = 0.0 if hit else -1.0                                                       <L 443>
            if (var_87) {
            }
            if (!var_87) {
            }
            var_93 = wp::where(var_87, var_91, var_92);
        }
        var_94 = wp::where(var_77, var_93, var_67);
        // if wp.static(int(GeomType.FLEX) in geom_ray_types):                                <L 455>
        // if cull_backfaces and d >= 0.0 and wp.dot(ray_dir_world, n) > 0.0:                 <L 496>
        var_96 = var_cull_backfaces;
        if (var_96) {
            var_98 = (var_94 >= var_97);
            var_96 = var_96 && var_98;
        }
        if (var_96) {
            var_99 = wp::dot(var_ray_dir_world, var_68);
            var_101 = (var_99 > var_100);
            var_96 = var_96 && var_101;
        }
        if (var_96) {
            // d = -1.0                                                                       <L 497>
        }
        var_103 = wp::where(var_96, var_102, var_94);
        // if wp.static(first_hit):                                                           <L 499>
        // if d >= 0.0 and d < dist:                                                          <L 501>
        var_107 = (var_103 >= var_106);
        var_105 = var_107;
        if (var_105) {
            var_108 = (var_103 < var_0);
            var_105 = var_105 && var_108;
        }
        if (var_105) {
            // return hit_geom_id, d, n, u, v, f, hit_mesh_id                                 <L 502>
            ret_0 = var_50;
            ret_1 = var_103;
            ret_2 = var_68;
            ret_3 = var_28;
            ret_4 = var_30;
            ret_5 = var_32;
            ret_6 = var_26;
            return;
        }
    goto start_while_0;
    end_while_0:;
    // return geom_id, dist, normal, bary_u, bary_v, face_idx, geom_mesh_id                   <L 513>
    ret_0 = var_6;
    ret_1 = var_0;
    ret_2 = var_4;
    ret_3 = var_8;
    ret_4 = var_10;
    ret_5 = var_12;
    ret_6 = var_14;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:521
static CUDA_CALLABLE void _make_compute_lighting__locals__compute_lighting_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    bool var_use_shadows,
    wp::uint64 var_bvh_id,
    wp::int32 var_group_root,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_bvh_nflexgeom,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::int32 var_worldid,
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::array_t<wp::uint64> var_hfield_bvh_id,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::array_t<wp::uint64> var_flex_bvh_id,
    wp::array_t<wp::int32> var_flex_group_root,
    bool var_lightactive,
    wp::int32 var_lighttype,
    bool var_lightcastshadow,
    wp::vec_t<3, wp::float32> var_lightpos,
    wp::vec_t<3, wp::float32> var_lightdir,
    wp::vec_t<3, wp::float32> var_lightattenuation,
    wp::float32 var_lightcutoff_rad,
    wp::float32 var_lightexp,
    wp::vec_t<3, wp::float32> var_lightdiff,
    wp::vec_t<3, wp::float32> var_lightspec,
    wp::vec_t<3, wp::float32> var_normal,
    wp::vec_t<3, wp::float32> var_hitpoint,
    wp::vec_t<3, wp::float32> var_view_dir,
    wp::float32 var_mat_spec,
    wp::float32 var_mat_shin_exp,
    bool var_cull_backfaces,
    bool var_enable_specular,
    bool var_default_attenuation,
    bool var_has_spot,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    bool var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    const wp::float32 var_7 = 10000000000.0;
    wp::float32 var_8;
    const wp::float32 var_9 = 1.0;
    const wp::int32 var_10 = 1;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::float32 var_15;
    bool var_16;
    const wp::float32 var_17 = 1.0;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = 1.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    bool var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    bool var_29;
    const wp::float32 var_30 = 0.0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.0;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::float32 var_42 = 0.0;
    bool var_43;
    const wp::float32 var_44 = 1.0;
    bool var_45;
    const wp::float32 var_46 = 0.0001;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    const wp::float32 var_49 = 0.001;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    bool var_52;
    const wp::float32 var_53 = 100000000.0;
    wp::float32 var_54;
    wp::int32 var_55;
    wp::float32 var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::int32 var_60;
    wp::int32 var_61;
    const wp::int32 var_62 = -1;
    bool var_63;
    const wp::float32 var_64 = 0.3;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    bool var_71;
    const wp::float32 var_72 = 0.0;
    bool var_73;
    const wp::float32 var_74 = 0.0;
    bool var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    const wp::float32 var_78 = 0.0;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    wp::vec_t<3, wp::float32> var_86;
    //---------
    // forward
    // def compute_lighting(                                                                  <L 522>
    // diff_rgb = wp.vec3(0.0)                                                                <L 568>
    var_1 = wp::vec_t<3, wp::float32>(var_0);
    // spec_rgb = wp.vec3(0.0)                                                                <L 569>
    var_3 = wp::vec_t<3, wp::float32>(var_2);
    // if not lightactive:                                                                    <L 573>
    var_4 = wp::unot(var_lightactive);
    if (var_4) {
        // return diff_rgb, spec_rgb                                                          <L 574>
        ret_0 = var_1;
        ret_1 = var_3;
        return;
    }
    // L = wp.vec3(0.0)                                                                       <L 576>
    var_6 = wp::vec_t<3, wp::float32>(var_5);
    // dist_to_light = float(MJ_MAXVAL)                                                       <L 577>
    var_8 = wp::float(var_7);
    // attenuation = 1.0                                                                      <L 578>
    // if lighttype == 1:  # directional light                                                <L 580>
    var_11 = (var_lighttype == var_10);
    if (var_11) {
        // L = -lightdir                                                                      <L 582>
        var_12 = wp::neg(var_lightdir);
    }
    if (!var_11) {
        // L, dist_to_light = math.normalize_with_norm(lightpos - hitpoint)                   <L 584>
        var_13 = wp::sub(var_lightpos, var_hitpoint);
        normalize_with_norm_0(var_13, var_14, var_15);
        // if not default_attenuation:                                                        <L 585>
        var_16 = wp::unot(var_default_attenuation);
        if (var_16) {
            // light_attenuation_factor = wp.vec3(1.0, dist_to_light, dist_to_light * dist_to_light)       <L 586>
            var_18 = wp::mul(var_15, var_15);
            var_19 = wp::vec_t<3, wp::float32>(var_17, var_15, var_18);
            // attenuation = math.safe_div(1.0, wp.dot(light_attenuation_factor, lightattenuation))       <L 587>
            var_21 = wp::dot(var_19, var_lightattenuation);
            var_22 = safe_div_0(var_20, var_21);
        }
        var_23 = wp::where(var_16, var_22, var_9);
        // if has_spot:                                                                       <L 588>
        if (var_has_spot) {
            // if lighttype == 0:  # spot light                                               <L 589>
            var_25 = (var_lighttype == var_24);
            if (var_25) {
                // cos_theta = wp.dot(-L, lightdir)                                           <L 590>
                var_26 = wp::neg(var_14);
                var_27 = wp::dot(var_26, var_lightdir);
                // cos_cutoff = wp.cos(lightcutoff_rad)                                       <L 591>
                var_28 = wp::cos(var_lightcutoff_rad);
                // if cos_theta < cos_cutoff:                                                 <L 592>
                var_29 = (var_27 < var_28);
                if (var_29) {
                    // return diff_rgb, spec_rgb                                              <L 593>
                    ret_0 = var_1;
                    ret_1 = var_3;
                    return;
                }
                // attenuation = attenuation * wp.pow(wp.max(cos_theta, 0.0), lightexp)       <L 594>
                var_31 = wp::max(var_27, var_30);
                var_32 = wp::pow(var_31, var_lightexp);
                var_33 = wp::mul(var_23, var_32);
            }
            var_34 = wp::where(var_25, var_33, var_23);
        }
        var_35 = wp::where(var_has_spot, var_34, var_23);
    }
    var_36 = wp::where(var_11, var_12, var_14);
    var_37 = wp::where(var_11, var_8, var_15);
    var_38 = wp::where(var_11, var_9, var_35);
    // ndotl = wp.max(0.0, wp.dot(normal, L))                                                 <L 596>
    var_40 = wp::dot(var_normal, var_36);
    var_41 = wp::max(var_39, var_40);
    // if ndotl == 0.0:                                                                       <L 597>
    var_43 = (var_41 == var_42);
    if (var_43) {
        // return diff_rgb, spec_rgb                                                          <L 598>
        ret_0 = var_1;
        ret_1 = var_3;
        return;
    }
    // visible = 1.0                                                                          <L 600>
    // if use_shadows and lightcastshadow:                                                    <L 602>
    var_45 = var_use_shadows;
    if (var_45) {
        var_45 = var_45 && var_lightcastshadow;
    }
    if (var_45) {
        // shadow_origin = hitpoint + normal * 1.0e-4                                         <L 605>
        var_47 = wp::mul(var_normal, var_46);
        var_48 = wp::add(var_hitpoint, var_47);
        // max_t = dist_to_light - 1.0e-3                                                     <L 607>
        var_50 = wp::sub(var_37, var_49);
        // if lighttype == 1:  # directional light                                            <L 608>
        var_52 = (var_lighttype == var_51);
        if (var_52) {
            // max_t = 1.0e8                                                                  <L 609>
        }
        var_54 = wp::where(var_52, var_53, var_50);
        // shadow_geom_id, shadow_d, shadow_n, shadow_u, shadow_v, shadow_f, shadow_mesh_id = cast_ray_first_hit(       <L 611>
        // geom_type,                                                                         <L 612>
        // geom_dataid,                                                                       <L 613>
        // geom_size,                                                                         <L 614>
        // flex_vertadr,                                                                      <L 615>
        // flex_edge,                                                                         <L 616>
        // flex_radius,                                                                       <L 617>
        // geom_xpos_in,                                                                      <L 618>
        // geom_xmat_in,                                                                      <L 619>
        // flexvert_xpos_in,                                                                  <L 620>
        // bvh_id,                                                                            <L 621>
        // group_root,                                                                        <L 622>
        // worldid,                                                                           <L 623>
        // bvh_ngeom,                                                                         <L 624>
        // bvh_nflexgeom,                                                                     <L 625>
        // enabled_geom_ids,                                                                  <L 626>
        // mesh_bvh_id,                                                                       <L 627>
        // hfield_bvh_id,                                                                     <L 628>
        // flex_geom_flexid,                                                                  <L 629>
        // flex_geom_edgeid,                                                                  <L 630>
        // flex_bvh_id,                                                                       <L 631>
        // flex_group_root,                                                                   <L 632>
        // shadow_origin,                                                                     <L 633>
        // L,                                                                                 <L 634>
        // max_t,                                                                             <L 635>
        // cull_backfaces,                                                                    <L 636>
        _make_cast_ray__locals__cast_ray_1(var_geom_type, var_geom_dataid, var_geom_size, var_flex_vertadr, var_flex_edge, var_flex_radius, var_geom_xpos_in, var_geom_xmat_in, var_flexvert_xpos_in, var_bvh_id, var_group_root, var_worldid, var_bvh_ngeom, var_bvh_nflexgeom, var_enabled_geom_ids, var_mesh_bvh_id, var_hfield_bvh_id, var_flex_geom_flexid, var_flex_geom_edgeid, var_flex_bvh_id, var_flex_group_root, var_48, var_36, var_54, var_cull_backfaces, var_55, var_56, var_57, var_58, var_59, var_60, var_61);
        // if shadow_geom_id != -1:                                                           <L 639>
        var_63 = (var_55 != var_62);
        if (var_63) {
            // visible = NO_LIGHT_AMBIENT_FALLBACK                                            <L 640>
            var_65 = wp::copy(var_64);
        }
        var_66 = wp::where(var_63, var_65, var_44);
    }
    var_67 = wp::where(var_45, var_66, var_44);
    // weight = attenuation * visible                                                         <L 642>
    var_68 = wp::mul(var_38, var_67);
    // diff_rgb = lightdiff * (ndotl * weight)                                                <L 643>
    var_69 = wp::mul(var_41, var_68);
    var_70 = wp::mul(var_lightdiff, var_69);
    // if enable_specular:                                                                    <L 644>
    if (var_enable_specular) {
        // if mat_spec > 0.0 and mat_shin_exp > 0.0:                                          <L 645>
        var_73 = (var_mat_spec > var_72);
        var_71 = var_73;
        if (var_71) {
            var_75 = (var_mat_shin_exp > var_74);
            var_71 = var_71 && var_75;
        }
        if (var_71) {
            // H = wp.normalize(L + view_dir)                                                 <L 646>
            var_76 = wp::add(var_36, var_view_dir);
            var_77 = wp::normalize(var_76);
            // ndoth = wp.max(0.0, wp.dot(normal, H))                                         <L 647>
            var_79 = wp::dot(var_normal, var_77);
            var_80 = wp::max(var_78, var_79);
            // spec_rgb = lightspec * (mat_spec * wp.pow(ndoth, mat_shin_exp) * weight)       <L 648>
            var_81 = wp::pow(var_80, var_mat_shin_exp);
            var_82 = wp::mul(var_mat_spec, var_81);
            var_83 = wp::mul(var_82, var_68);
            var_84 = wp::mul(var_lightspec, var_83);
        }
        var_85 = wp::where(var_71, var_84, var_3);
    }
    var_86 = wp::where(var_enable_specular, var_85, var_3);
    // return diff_rgb, spec_rgb                                                              <L 650>
    ret_0 = var_70;
    ret_1 = var_86;
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:32
static CUDA_CALLABLE void adj__ray_map_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:213
static CUDA_CALLABLE void adj_ray_plane_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:690
static CUDA_CALLABLE void adj_ray_mesh_with_bvh_0(
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::int32 var_mesh_geom_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 var_max_t,
    bool var_cull_backfaces,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2,
    wp::float32 & ret_3,
    wp::int32 & ret_4,
    wp::int32 & ret_5,
    wp::array_t<wp::uint64> & adj_mesh_bvh_id,
    wp::int32 & adj_mesh_geom_id,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_max_t,
    bool & adj_cull_backfaces,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::float32 & adj_ret_2,
    wp::float32 & adj_ret_3,
    wp::int32 & adj_ret_4,
    wp::int32 & adj_ret_5)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:302
static CUDA_CALLABLE void adj__make_cast_ray__locals__cast_ray_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::uint64 var_bvh_id,
    wp::int32 var_group_root,
    wp::int32 var_worldid,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_flex_bvh_ngeom,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::array_t<wp::uint64> var_hfield_bvh_id,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::array_t<wp::uint64> var_flex_bvh_id,
    wp::array_t<wp::int32> var_flex_group_root,
    wp::vec_t<3, wp::float32> var_ray_origin_world,
    wp::vec_t<3, wp::float32> var_ray_dir_world,
    wp::float32 var_max_dist,
    bool var_cull_backfaces,
    wp::int32 & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & ret_3,
    wp::float32 & ret_4,
    wp::int32 & ret_5,
    wp::int32 & ret_6,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_flex_edge,
    wp::array_t<wp::float32> & adj_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_flexvert_xpos_in,
    wp::uint64 & adj_bvh_id,
    wp::int32 & adj_group_root,
    wp::int32 & adj_worldid,
    wp::int32 & adj_bvh_ngeom,
    wp::int32 & adj_flex_bvh_ngeom,
    wp::array_t<wp::int32> & adj_enabled_geom_ids,
    wp::array_t<wp::uint64> & adj_mesh_bvh_id,
    wp::array_t<wp::uint64> & adj_hfield_bvh_id,
    wp::array_t<wp::int32> & adj_flex_geom_flexid,
    wp::array_t<wp::int32> & adj_flex_geom_edgeid,
    wp::array_t<wp::uint64> & adj_flex_bvh_id,
    wp::array_t<wp::int32> & adj_flex_group_root,
    wp::vec_t<3, wp::float32> & adj_ray_origin_world,
    wp::vec_t<3, wp::float32> & adj_ray_dir_world,
    wp::float32 & adj_max_dist,
    bool & adj_cull_backfaces,
    wp::int32 & adj_ret_0,
    wp::float32 & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::float32 & adj_ret_3,
    wp::float32 & adj_ret_4,
    wp::int32 & adj_ret_5,
    wp::int32 & adj_ret_6)
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
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    wp::adj_div(var_x, var_4, var_5, adj_x, adj_4, adj_5);
    wp::adj_where(var_1, var_y, var_3, adj_1, adj_y, adj_3, adj_4);
    // adj: return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                 <L 2>
    // adj: def safe_div(x: Any, y: Any) -> Any:                                              <L 1>
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:222
static CUDA_CALLABLE void adj_sample_skybox_0(
    wp::texture2d_t var_skybox_tex,
    wp::float32 var_face_width_inv,
    wp::vec_t<3, wp::float32> var_ray_dir_world,
    wp::texture2d_t & adj_skybox_tex,
    wp::float32 & adj_face_width_inv,
    wp::vec_t<3, wp::float32> & adj_ray_dir_world,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/se-semantic-recovery/render-parity-df80c9c5-20260928/source-v12/stable_envs/render/mjwarp.py:755
static CUDA_CALLABLE void adj_gl_rounding__locals__pack_rgba_round_0(
    wp::float32 var_r,
    wp::float32 var_g,
    wp::float32 var_b,
    wp::float32 var_a,
    wp::float32 & adj_r,
    wp::float32 & adj_g,
    wp::float32 & adj_b,
    wp::float32 & adj_a,
    wp::uint32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:172
static CUDA_CALLABLE void adj_sample_texture_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_mesh_faceadr,
    wp::int32 var_geom_id,
    wp::vec_t<2, wp::float32> var_tex_repeat,
    wp::texture2d_t var_tex,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_rot,
    wp::array_t<wp::vec_t<3, wp::int32>> var_mesh_facetexcoord,
    wp::array_t<wp::vec_t<2, wp::float32>> var_mesh_texcoord,
    wp::array_t<wp::int32> var_mesh_texcoord_offsets,
    wp::vec_t<3, wp::float32> var_hit_point,
    wp::float32 var_bary_u,
    wp::float32 var_bary_v,
    wp::int32 var_f,
    wp::int32 var_mesh_id,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::int32> & adj_mesh_faceadr,
    wp::int32 & adj_geom_id,
    wp::vec_t<2, wp::float32> & adj_tex_repeat,
    wp::texture2d_t & adj_tex,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_mesh_facetexcoord,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_mesh_texcoord,
    wp::array_t<wp::int32> & adj_mesh_texcoord_offsets,
    wp::vec_t<3, wp::float32> & adj_hit_point,
    wp::float32 & adj_bary_u,
    wp::float32 & adj_bary_v,
    wp::int32 & adj_f,
    wp::int32 & adj_mesh_id,
    wp::vec_t<3, wp::float32> & adj_ret)
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
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    bool adj_2 = {};
    wp::float32 adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
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
        goto label0;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    goto label1;
    //---------
    // reverse
    label1:;
    adj_0 += adj_ret_1;
    adj_4 += adj_ret_0;
    wp::adj_div(var_x, var_0, adj_x, adj_0, adj_4);
    // adj: return x / norm, norm                                                             <L 5>
    if (var_2) {
        label0:;
        adj_3 += adj_ret_1;
        adj_x += adj_ret_0;
        // adj: return x, 0.0                                                                 <L 4>
    }
    // adj: if norm == 0.0:                                                                   <L 3>
    wp::adj_length(var_x, var_0, adj_x, adj_0);
    // adj: norm = wp.length(x)                                                               <L 2>
    // adj: def normalize_with_norm(x: Any):                                                  <L 1>
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/ray.py:732
static CUDA_CALLABLE void adj_ray_mesh_with_bvh_anyhit_0(
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::int32 var_mesh_geom_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 var_max_t,
    wp::array_t<wp::uint64> & adj_mesh_bvh_id,
    wp::int32 & adj_mesh_geom_id,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_max_t,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:302
static CUDA_CALLABLE void adj__make_cast_ray__locals__cast_ray_1(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::uint64 var_bvh_id,
    wp::int32 var_group_root,
    wp::int32 var_worldid,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_flex_bvh_ngeom,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::array_t<wp::uint64> var_hfield_bvh_id,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::array_t<wp::uint64> var_flex_bvh_id,
    wp::array_t<wp::int32> var_flex_group_root,
    wp::vec_t<3, wp::float32> var_ray_origin_world,
    wp::vec_t<3, wp::float32> var_ray_dir_world,
    wp::float32 var_max_dist,
    bool var_cull_backfaces,
    wp::int32 & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & ret_3,
    wp::float32 & ret_4,
    wp::int32 & ret_5,
    wp::int32 & ret_6,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_flex_edge,
    wp::array_t<wp::float32> & adj_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_flexvert_xpos_in,
    wp::uint64 & adj_bvh_id,
    wp::int32 & adj_group_root,
    wp::int32 & adj_worldid,
    wp::int32 & adj_bvh_ngeom,
    wp::int32 & adj_flex_bvh_ngeom,
    wp::array_t<wp::int32> & adj_enabled_geom_ids,
    wp::array_t<wp::uint64> & adj_mesh_bvh_id,
    wp::array_t<wp::uint64> & adj_hfield_bvh_id,
    wp::array_t<wp::int32> & adj_flex_geom_flexid,
    wp::array_t<wp::int32> & adj_flex_geom_edgeid,
    wp::array_t<wp::uint64> & adj_flex_bvh_id,
    wp::array_t<wp::int32> & adj_flex_group_root,
    wp::vec_t<3, wp::float32> & adj_ray_origin_world,
    wp::vec_t<3, wp::float32> & adj_ray_dir_world,
    wp::float32 & adj_max_dist,
    bool & adj_cull_backfaces,
    wp::int32 & adj_ret_0,
    wp::float32 & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::float32 & adj_ret_3,
    wp::float32 & adj_ret_4,
    wp::int32 & adj_ret_5,
    wp::int32 & adj_ret_6)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render.py:521
static CUDA_CALLABLE void adj__make_compute_lighting__locals__compute_lighting_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    bool var_use_shadows,
    wp::uint64 var_bvh_id,
    wp::int32 var_group_root,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_bvh_nflexgeom,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::int32 var_worldid,
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::array_t<wp::uint64> var_hfield_bvh_id,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::array_t<wp::uint64> var_flex_bvh_id,
    wp::array_t<wp::int32> var_flex_group_root,
    bool var_lightactive,
    wp::int32 var_lighttype,
    bool var_lightcastshadow,
    wp::vec_t<3, wp::float32> var_lightpos,
    wp::vec_t<3, wp::float32> var_lightdir,
    wp::vec_t<3, wp::float32> var_lightattenuation,
    wp::float32 var_lightcutoff_rad,
    wp::float32 var_lightexp,
    wp::vec_t<3, wp::float32> var_lightdiff,
    wp::vec_t<3, wp::float32> var_lightspec,
    wp::vec_t<3, wp::float32> var_normal,
    wp::vec_t<3, wp::float32> var_hitpoint,
    wp::vec_t<3, wp::float32> var_view_dir,
    wp::float32 var_mat_spec,
    wp::float32 var_mat_shin_exp,
    bool var_cull_backfaces,
    bool var_enable_specular,
    bool var_default_attenuation,
    bool var_has_spot,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_flex_edge,
    wp::array_t<wp::float32> & adj_flex_radius,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_flexvert_xpos_in,
    bool & adj_use_shadows,
    wp::uint64 & adj_bvh_id,
    wp::int32 & adj_group_root,
    wp::int32 & adj_bvh_ngeom,
    wp::int32 & adj_bvh_nflexgeom,
    wp::array_t<wp::int32> & adj_enabled_geom_ids,
    wp::int32 & adj_worldid,
    wp::array_t<wp::uint64> & adj_mesh_bvh_id,
    wp::array_t<wp::uint64> & adj_hfield_bvh_id,
    wp::array_t<wp::int32> & adj_flex_geom_flexid,
    wp::array_t<wp::int32> & adj_flex_geom_edgeid,
    wp::array_t<wp::uint64> & adj_flex_bvh_id,
    wp::array_t<wp::int32> & adj_flex_group_root,
    bool & adj_lightactive,
    wp::int32 & adj_lighttype,
    bool & adj_lightcastshadow,
    wp::vec_t<3, wp::float32> & adj_lightpos,
    wp::vec_t<3, wp::float32> & adj_lightdir,
    wp::vec_t<3, wp::float32> & adj_lightattenuation,
    wp::float32 & adj_lightcutoff_rad,
    wp::float32 & adj_lightexp,
    wp::vec_t<3, wp::float32> & adj_lightdiff,
    wp::vec_t<3, wp::float32> & adj_lightspec,
    wp::vec_t<3, wp::float32> & adj_normal,
    wp::vec_t<3, wp::float32> & adj_hitpoint,
    wp::vec_t<3, wp::float32> & adj_view_dir,
    wp::float32 & adj_mat_spec,
    wp::float32 & adj_mat_shin_exp,
    bool & adj_cull_backfaces,
    bool & adj_enable_specular,
    bool & adj_default_attenuation,
    bool & adj_has_spot,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void render__locals___render_megakernel_01f6092e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::int32> var_geom_matid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::vec_t<4, wp::float32>> var_geom_rgba,
    wp::array_t<wp::int32> var_cam_projection,
    wp::array_t<wp::float32> var_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::float32>> var_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> var_cam_intrinsic,
    wp::array_t<wp::int32> var_light_type,
    wp::array_t<bool> var_light_castshadow,
    wp::array_t<bool> var_light_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_attenuation,
    wp::array_t<wp::float32> var_light_cutoff,
    wp::array_t<wp::float32> var_light_exponent,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_ambient,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_diffuse,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_specular,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_radius,
    wp::array_t<wp::int32> var_mesh_faceadr,
    wp::array_t<wp::int32> var_mat_texid,
    wp::array_t<wp::vec_t<2, wp::float32>> var_mat_texrepeat,
    wp::array_t<wp::float32> var_mat_emission,
    wp::array_t<wp::float32> var_mat_specular,
    wp::array_t<wp::float32> var_mat_shininess,
    wp::array_t<wp::vec_t<4, wp::float32>> var_mat_rgba,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::int32 var_nrender,
    bool var_use_shadows,
    wp::int32 var_bvh_ngeom,
    wp::int32 var_bvh_nflexgeom,
    wp::array_t<wp::vec_t<2, wp::int32>> var_cam_res,
    wp::array_t<wp::int32> var_cam_id_map,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray,
    wp::array_t<wp::int32> var_rgb_adr,
    wp::array_t<wp::int32> var_depth_adr,
    wp::array_t<wp::int32> var_seg_adr,
    wp::array_t<bool> var_render_rgb,
    wp::array_t<bool> var_render_depth,
    wp::array_t<bool> var_render_seg,
    wp::uint64 var_bvh_id,
    wp::array_t<wp::int32> var_group_root,
    wp::array_t<wp::uint64> var_flex_bvh_id,
    wp::array_t<wp::int32> var_flex_group_root,
    wp::array_t<wp::int32> var_enabled_geom_ids,
    wp::array_t<wp::uint64> var_mesh_bvh_id,
    wp::array_t<wp::vec_t<3, wp::int32>> var_mesh_facetexcoord,
    wp::array_t<wp::vec_t<2, wp::float32>> var_mesh_texcoord,
    wp::array_t<wp::int32> var_mesh_texcoord_offsets,
    wp::array_t<wp::uint64> var_hfield_bvh_id,
    wp::array_t<wp::vec_t<4, wp::float32>> var_flex_rgba,
    wp::array_t<wp::int32> var_flex_geom_flexid,
    wp::array_t<wp::int32> var_flex_geom_edgeid,
    wp::array_t<wp::int32> var_skybox_tex_id,
    wp::array_t<wp::int32> var_skybox_face_width,
    wp::array_t<wp::texture2d_t> var_textures,
    wp::array_t<wp::vec_t<3, wp::float32>> var_splat_position,
    wp::array_t<wp::quat_t<wp::float32>> var_splat_rotation,
    wp::array_t<wp::vec_t<3, wp::float32>> var_splat_scale,
    wp::array_t<wp::vec_t<4, wp::float32>> var_splat_rgba,
    wp::uint64 var_splat_bvh_id,
    wp::array_t<wp::int32> var_splat_group_root,
    wp::array_t<wp::uint32> var_rgb_out,
    wp::array_t<wp::float32> var_depth_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_seg_out)
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
        const wp::int32 var_2 = -1;
        wp::int32 var_3;
        const wp::int32 var_4 = -1;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::vec_t<2, wp::int32>* var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::vec_t<2, wp::int32> var_13;
        wp::vec_t<2, wp::int32>* var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::vec_t<2, wp::int32> var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        bool var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = -1;
        bool var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        bool var_29;
        bool* var_30;
        bool var_31;
        bool var_32;
        bool* var_33;
        bool var_34;
        bool var_35;
        bool* var_36;
        bool var_37;
        bool var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const bool var_42 = true;
        wp::vec_t<3, wp::float32>* var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::mat_t<3, 3, wp::float32>* var_49;
        wp::mat_t<3, 3, wp::float32> var_50;
        wp::mat_t<3, 3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::int32* var_53;
        const wp::float32 var_54 = 10000000000.0;
        wp::float32 var_55;
        const bool var_56 = true;
        wp::int32 var_57;
        wp::float32 var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::int32 var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        const wp::float32 var_65 = 0.0;
        wp::vec_t<3, wp::float32> var_66;
        const wp::float32 var_67 = 1.0;
        wp::float32 var_68;
        const wp::float32 var_69 = -1.0;
        wp::float32 var_70;
        const bool var_71 = false;
        bool var_72;
        bool* var_73;
        bool var_74;
        const wp::int32 var_75 = -1;
        bool var_76;
        const wp::int32 var_77 = -2;
        bool var_78;
        const wp::int32 var_79 = 9;
        const wp::int32 var_80 = 9;
        wp::int32 var_81;
        wp::vec_t<2, wp::int32> var_82;
        wp::int32* var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 5;
        const wp::int32 var_87 = 5;
        wp::int32 var_88;
        wp::vec_t<2, wp::int32> var_89;
        wp::int32* var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        const wp::int32 var_93 = -1;
        bool var_94;
        bool* var_95;
        bool var_96;
        const wp::float32 var_97 = 0.0;
        const bool var_98 = false;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        bool var_102;
        bool var_103;
        const bool var_104 = true;
        bool* var_105;
        bool var_106;
        wp::shape_t* var_107;
        const wp::int32 var_108 = 0;
        wp::int32 var_109;
        wp::shape_t var_110;
        wp::int32 var_111;
        wp::int32* var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::texture2d_t* var_115;
        const wp::float32 var_116 = 1.0;
        wp::shape_t* var_117;
        const wp::int32 var_118 = 0;
        wp::int32 var_119;
        wp::shape_t var_120;
        wp::int32 var_121;
        wp::int32* var_122;
        wp::float32 var_123;
        wp::int32 var_124;
        wp::float32 var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::texture2d_t var_127;
        const bool var_128 = false;
        const wp::int32 var_129 = 0;
        wp::float32 var_130;
        const wp::float32 var_131 = 255.0;
        wp::float32 var_132;
        const wp::int32 var_133 = 1;
        wp::float32 var_134;
        const wp::float32 var_135 = 255.0;
        wp::float32 var_136;
        const wp::int32 var_137 = 2;
        wp::float32 var_138;
        const wp::float32 var_139 = 255.0;
        wp::float32 var_140;
        const wp::float32 var_141 = 255.0;
        wp::uint32 var_142;
        wp::int32* var_143;
        wp::int32 var_144;
        wp::int32 var_145;
        bool* var_146;
        bool var_147;
        const bool var_148 = false;
        const wp::uint32 var_149 = 4279834931u;
        wp::int32* var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        bool var_153;
        bool* var_154;
        bool var_155;
        wp::float32 var_156;
        const bool var_157 = false;
        const wp::int32 var_158 = 2;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::int32* var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        bool var_165;
        wp::float32 var_166;
        bool var_167;
        bool* var_168;
        bool var_169;
        bool var_170;
        wp::vec_t<3, wp::float32> var_171;
        wp::vec_t<3, wp::float32> var_172;
        const wp::int32 var_173 = -2;
        bool var_174;
        wp::vec_t<4, wp::float32>* var_175;
        wp::vec_t<4, wp::float32> var_176;
        wp::vec_t<4, wp::float32> var_177;
        wp::shape_t* var_178;
        const wp::int32 var_179 = 0;
        wp::int32 var_180;
        wp::shape_t var_181;
        wp::int32 var_182;
        wp::int32* var_183;
        const wp::int32 var_184 = -1;
        bool var_185;
        wp::int32 var_186;
        wp::shape_t* var_187;
        const wp::int32 var_188 = 0;
        wp::int32 var_189;
        wp::shape_t var_190;
        wp::int32 var_191;
        wp::vec_t<4, wp::float32>* var_192;
        wp::vec_t<4, wp::float32> var_193;
        wp::vec_t<4, wp::float32> var_194;
        wp::shape_t* var_195;
        const wp::int32 var_196 = 0;
        wp::int32 var_197;
        wp::shape_t var_198;
        wp::int32 var_199;
        wp::shape_t* var_200;
        const wp::int32 var_201 = 0;
        wp::int32 var_202;
        wp::shape_t var_203;
        wp::int32 var_204;
        wp::int32* var_205;
        wp::vec_t<4, wp::float32>* var_206;
        wp::int32 var_207;
        wp::vec_t<4, wp::float32> var_208;
        wp::vec_t<4, wp::float32> var_209;
        wp::vec_t<4, wp::float32> var_210;
        wp::vec_t<4, wp::float32> var_211;
        const wp::int32 var_212 = 0;
        wp::float32 var_213;
        const wp::int32 var_214 = 1;
        wp::float32 var_215;
        const wp::int32 var_216 = 2;
        wp::float32 var_217;
        wp::vec_t<3, wp::float32> var_218;
        const bool var_219 = true;
        const wp::int32 var_220 = -2;
        bool var_221;
        wp::shape_t* var_222;
        const wp::int32 var_223 = 0;
        wp::int32 var_224;
        wp::shape_t var_225;
        wp::int32 var_226;
        wp::int32* var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        const wp::int32 var_230 = 0;
        bool var_231;
        wp::shape_t* var_232;
        const wp::int32 var_233 = 0;
        wp::int32 var_234;
        wp::shape_t var_235;
        wp::int32 var_236;
        const wp::int32 var_237 = 1;
        wp::int32* var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        const wp::int32 var_241 = 0;
        bool var_242;
        wp::shape_t* var_243;
        const wp::int32 var_244 = 0;
        wp::int32 var_245;
        wp::shape_t var_246;
        wp::int32 var_247;
        wp::vec_t<2, wp::float32>* var_248;
        wp::texture2d_t* var_249;
        wp::vec_t<3, wp::float32>* var_250;
        wp::mat_t<3, 3, wp::float32>* var_251;
        wp::vec_t<3, wp::float32> var_252;
        wp::vec_t<2, wp::float32> var_253;
        wp::texture2d_t var_254;
        wp::vec_t<3, wp::float32> var_255;
        wp::mat_t<3, 3, wp::float32> var_256;
        wp::vec_t<3, wp::float32> var_257;
        wp::vec_t<3, wp::float32> var_258;
        wp::vec_t<3, wp::float32> var_259;
        wp::vec_t<3, wp::float32> var_260;
        const wp::float32 var_261 = 0.5;
        wp::float32 var_262;
        const wp::float32 var_263 = 64.0;
        wp::float32 var_264;
        const wp::float32 var_265 = 0.0;
        wp::float32 var_266;
        const bool var_267 = true;
        const wp::int32 var_268 = -2;
        bool var_269;
        wp::shape_t* var_270;
        const wp::int32 var_271 = 0;
        wp::int32 var_272;
        wp::shape_t var_273;
        wp::int32 var_274;
        wp::int32* var_275;
        wp::int32 var_276;
        wp::int32 var_277;
        const wp::int32 var_278 = 0;
        bool var_279;
        const bool var_280 = true;
        wp::shape_t* var_281;
        const wp::int32 var_282 = 0;
        wp::int32 var_283;
        wp::shape_t var_284;
        wp::int32 var_285;
        wp::float32* var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        wp::shape_t* var_289;
        const wp::int32 var_290 = 0;
        wp::int32 var_291;
        wp::shape_t var_292;
        wp::int32 var_293;
        wp::float32* var_294;
        const wp::float32 var_295 = 128.0;
        wp::float32 var_296;
        wp::float32 var_297;
        const bool var_298 = true;
        wp::shape_t* var_299;
        const wp::int32 var_300 = 0;
        wp::int32 var_301;
        wp::shape_t var_302;
        wp::int32 var_303;
        wp::float32* var_304;
        wp::float32 var_305;
        wp::float32 var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        const wp::float32 var_313 = 0.0;
        wp::vec_t<3, wp::float32> var_314;
        const bool var_315 = true;
        wp::vec_t<3, wp::float32> var_316;
        const bool var_317 = true;
        const bool var_318 = true;
        const wp::vec_t<3, wp::float32> var_319 = wp::initializer_array<3,wp::float32>{0.10000000149011612, 0.10000000149011612, 0.10000000149011612};
        wp::vec_t<3, wp::float32> var_320;
        wp::vec_t<3, wp::float32> var_321;
        const bool var_322 = true;
        const wp::int32 var_323 = 0;
        wp::shape_t* var_324;
        const wp::int32 var_325 = 0;
        wp::int32 var_326;
        wp::shape_t var_327;
        wp::int32 var_328;
        bool* var_329;
        bool var_330;
        wp::shape_t* var_331;
        const wp::int32 var_332 = 0;
        wp::int32 var_333;
        wp::shape_t var_334;
        wp::int32 var_335;
        wp::vec_t<3, wp::float32>* var_336;
        wp::vec_t<3, wp::float32> var_337;
        wp::vec_t<3, wp::float32> var_338;
        wp::vec_t<3, wp::float32> var_339;
        bool var_340;
        wp::vec_t<3, wp::float32> var_341;
        bool var_342;
        const wp::int32 var_343 = 1;
        wp::shape_t* var_344;
        const wp::int32 var_345 = 0;
        wp::int32 var_346;
        wp::shape_t var_347;
        wp::int32 var_348;
        bool* var_349;
        bool var_350;
        wp::shape_t* var_351;
        const wp::int32 var_352 = 0;
        wp::int32 var_353;
        wp::shape_t var_354;
        wp::int32 var_355;
        wp::vec_t<3, wp::float32>* var_356;
        wp::vec_t<3, wp::float32> var_357;
        wp::vec_t<3, wp::float32> var_358;
        wp::vec_t<3, wp::float32> var_359;
        bool var_360;
        wp::vec_t<3, wp::float32> var_361;
        bool var_362;
        wp::vec_t<3, wp::float32> var_363;
        wp::shape_t* var_364;
        const wp::int32 var_365 = 0;
        wp::int32 var_366;
        wp::shape_t var_367;
        wp::int32 var_368;
        wp::array_t<wp::float32> var_369;
        wp::shape_t* var_370;
        const wp::int32 var_371 = 0;
        wp::int32 var_372;
        wp::shape_t var_373;
        wp::int32 var_374;
        wp::array_t<bool> var_375;
        wp::shape_t* var_376;
        const wp::int32 var_377 = 0;
        wp::int32 var_378;
        wp::shape_t var_379;
        wp::int32 var_380;
        wp::array_t<wp::int32> var_381;
        wp::shape_t* var_382;
        const wp::int32 var_383 = 0;
        wp::int32 var_384;
        wp::shape_t var_385;
        wp::int32 var_386;
        wp::array_t<bool> var_387;
        wp::array_t<wp::vec_t<3, wp::float32>> var_388;
        wp::array_t<wp::vec_t<3, wp::float32>> var_389;
        wp::shape_t* var_390;
        const wp::int32 var_391 = 0;
        wp::int32 var_392;
        wp::shape_t var_393;
        wp::int32 var_394;
        wp::array_t<wp::vec_t<3, wp::float32>> var_395;
        wp::shape_t* var_396;
        const wp::int32 var_397 = 0;
        wp::int32 var_398;
        wp::shape_t var_399;
        wp::int32 var_400;
        wp::array_t<wp::float32> var_401;
        wp::shape_t* var_402;
        const wp::int32 var_403 = 0;
        wp::int32 var_404;
        wp::shape_t var_405;
        wp::int32 var_406;
        wp::array_t<wp::vec_t<3, wp::float32>> var_407;
        wp::shape_t* var_408;
        const wp::int32 var_409 = 0;
        wp::int32 var_410;
        wp::shape_t var_411;
        wp::int32 var_412;
        wp::array_t<wp::vec_t<3, wp::float32>> var_413;
        const wp::int32 var_414 = 0;
        wp::int32* var_415;
        bool* var_416;
        wp::int32* var_417;
        bool* var_418;
        wp::vec_t<3, wp::float32>* var_419;
        wp::vec_t<3, wp::float32>* var_420;
        wp::vec_t<3, wp::float32>* var_421;
        wp::float32* var_422;
        const wp::float32 var_423 = 0.017453292519943295;
        wp::float32 var_424;
        wp::float32 var_425;
        wp::float32* var_426;
        wp::vec_t<3, wp::float32>* var_427;
        wp::vec_t<3, wp::float32>* var_428;
        const bool var_429 = true;
        const bool var_430 = true;
        const bool var_431 = true;
        const bool var_432 = true;
        wp::vec_t<3, wp::float32> var_433;
        wp::vec_t<3, wp::float32> var_434;
        wp::int32 var_435;
        bool var_436;
        wp::int32 var_437;
        bool var_438;
        wp::vec_t<3, wp::float32> var_439;
        wp::vec_t<3, wp::float32> var_440;
        wp::vec_t<3, wp::float32> var_441;
        wp::float32 var_442;
        wp::vec_t<3, wp::float32> var_443;
        wp::vec_t<3, wp::float32> var_444;
        wp::vec_t<3, wp::float32> var_445;
        wp::vec_t<3, wp::float32> var_446;
        wp::vec_t<3, wp::float32> var_447;
        const wp::int32 var_448 = 1;
        wp::int32* var_449;
        bool* var_450;
        wp::int32* var_451;
        bool* var_452;
        wp::vec_t<3, wp::float32>* var_453;
        wp::vec_t<3, wp::float32>* var_454;
        wp::vec_t<3, wp::float32>* var_455;
        wp::float32* var_456;
        const wp::float32 var_457 = 0.017453292519943295;
        wp::float32 var_458;
        wp::float32 var_459;
        wp::float32* var_460;
        wp::vec_t<3, wp::float32>* var_461;
        wp::vec_t<3, wp::float32>* var_462;
        const bool var_463 = true;
        const bool var_464 = true;
        const bool var_465 = true;
        const bool var_466 = true;
        wp::vec_t<3, wp::float32> var_467;
        wp::vec_t<3, wp::float32> var_468;
        wp::int32 var_469;
        bool var_470;
        wp::int32 var_471;
        bool var_472;
        wp::vec_t<3, wp::float32> var_473;
        wp::vec_t<3, wp::float32> var_474;
        wp::vec_t<3, wp::float32> var_475;
        wp::float32 var_476;
        wp::vec_t<3, wp::float32> var_477;
        wp::vec_t<3, wp::float32> var_478;
        wp::vec_t<3, wp::float32> var_479;
        wp::vec_t<3, wp::float32> var_480;
        wp::vec_t<3, wp::float32> var_481;
        const bool var_482 = true;
        wp::vec_t<3, wp::float32> var_483;
        wp::slice_t var_484;
        const wp::int32 var_485 = 0;
        const wp::int32 var_486 = 3;
        const wp::int32 var_487 = 1;
        const wp::int32 var_488 = 2;
        wp::vec_t<3, wp::float32> var_489;
        wp::vec_t<3, wp::float32> var_490;
        wp::int32* var_491;
        const bool var_492 = true;
        const wp::int32 var_493 = 1;
        const bool var_494 = false;
        const wp::float32 var_495 = 1.0;
        const wp::float32 var_496 = 0.0;
        const wp::float32 var_497 = 0.0;
        wp::vec_t<3, wp::float32> var_498;
        const wp::float32 var_499 = 0.0;
        const wp::float32 var_500 = 0.0;
        const wp::vec_t<3, wp::float32> var_501 = wp::initializer_array<3,wp::float32>{0.6000000238418579, 0.6000000238418579, 0.6000000238418579};
        const wp::vec_t<3, wp::float32> var_502 = wp::initializer_array<3,wp::float32>{0.0, 0.0, 0.0};
        const bool var_503 = true;
        const bool var_504 = true;
        const bool var_505 = true;
        const bool var_506 = false;
        wp::vec_t<3, wp::float32> var_507;
        wp::vec_t<3, wp::float32> var_508;
        wp::int32 var_509;
        wp::vec_t<3, wp::float32> var_510;
        wp::vec_t<3, wp::float32> var_511;
        wp::vec_t<3, wp::float32> var_512;
        const wp::float32 var_513 = 1.0;
        const wp::float32 var_514 = 1.0;
        const wp::float32 var_515 = 1.0;
        wp::vec_t<3, wp::float32> var_516;
        wp::vec_t<3, wp::float32> var_517;
        const wp::float32 var_518 = 0.0;
        const wp::float32 var_519 = 0.0;
        const wp::float32 var_520 = 0.0;
        wp::vec_t<3, wp::float32> var_521;
        wp::vec_t<3, wp::float32> var_522;
        const bool var_523 = false;
        const wp::int32 var_524 = 0;
        wp::float32 var_525;
        const wp::float32 var_526 = 255.0;
        wp::float32 var_527;
        const wp::int32 var_528 = 1;
        wp::float32 var_529;
        const wp::float32 var_530 = 255.0;
        wp::float32 var_531;
        const wp::int32 var_532 = 2;
        wp::float32 var_533;
        const wp::float32 var_534 = 255.0;
        wp::float32 var_535;
        const wp::float32 var_536 = 255.0;
        wp::uint32 var_537;
        wp::int32* var_538;
        wp::int32 var_539;
        wp::int32 var_540;
        //---------
        // forward
        // def _render_megakernel(                                                                <L 682>
        // worldid, rayid = wp.tid()                                                              <L 761>
        builtin_tid2d(var_0, var_1);
        // camid = int(-1)                                                                        <L 764>
        var_3 = wp::int(var_2);
        // rayid_local = int(-1)                                                                  <L 765>
        var_5 = wp::int(var_4);
        // accum = int(0)                                                                         <L 766>
        var_7 = wp::int(var_6);
        // for i in range(nrender):                                                               <L 767>
        var_8 = wp::range(var_nrender);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // num_i = cam_res[i][0] * cam_res[i][1]                                              <L 768>
            var_10 = wp::address(var_cam_res, var_9);
            var_13 = wp::load(var_10);
            var_12 = wp::extract(var_13, var_11);
            var_14 = wp::address(var_cam_res, var_9);
            var_17 = wp::load(var_14);
            var_16 = wp::extract(var_17, var_15);
            var_18 = wp::mul(var_12, var_16);
            // if rayid < accum + num_i:                                                          <L 769>
            var_19 = wp::add(var_7, var_18);
            var_20 = (var_1 < var_19);
            if (var_20) {
                // camid = i                                                                      <L 770>
                var_21 = wp::copy(var_9);
                // rayid_local = rayid - accum                                                    <L 771>
                var_22 = wp::sub(var_1, var_7);
                // break                                                                          <L 772>
                wp::assign(var_3, var_21);
                wp::assign(var_5, var_22);
                goto end_for_0;
            }
            // accum += num_i                                                                     <L 773>
            var_23 = wp::add(var_7, var_18);
            wp::assign(var_7, var_23);
            goto start_for_0;
        end_for_0:;
        // if camid == -1 or rayid_local < 0:                                                     <L 774>
        var_26 = (var_3 == var_25);
        var_24 = var_26;
        if (!var_24) {
            var_28 = (var_5 < var_27);
            var_24 = var_24 || var_28;
        }
        if (var_24) {
            // return                                                                             <L 775>
            return;
        }
        // if not render_rgb[camid] and not render_depth[camid] and not render_seg[camid]:        <L 777>
        var_30 = wp::address(var_render_rgb, var_3);
        var_32 = wp::load(var_30);
        var_31 = wp::unot(var_32);
        var_29 = var_31;
        if (var_29) {
            var_33 = wp::address(var_render_depth, var_3);
            var_35 = wp::load(var_33);
            var_34 = wp::unot(var_35);
            var_29 = var_29 && var_34;
        }
        if (var_29) {
            var_36 = wp::address(var_render_seg, var_3);
            var_38 = wp::load(var_36);
            var_37 = wp::unot(var_38);
            var_29 = var_29 && var_37;
        }
        if (var_29) {
            // return                                                                             <L 778>
            return;
        }
        // mujoco_cam_id = cam_id_map[camid]                                                      <L 781>
        var_39 = wp::address(var_cam_id_map, var_3);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // if wp.static(rc_static["use_precomputed_rays"]):                                       <L 783>
        // ray_dir_local_cam = ray[rayid]                                                         <L 784>
        var_43 = wp::address(var_ray, var_1);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // ray_origin_world = cam_xpos_in[worldid, mujoco_cam_id]                                 <L 802>
        var_46 = wp::address(var_cam_xpos_in, var_0, var_40);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // cam_mat_world = cam_xmat_in[worldid, mujoco_cam_id]                                    <L 803>
        var_49 = wp::address(var_cam_xmat_in, var_0, var_40);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // ray_dir_world = cam_mat_world @ ray_dir_local_cam                                      <L 804>
        var_52 = wp::mul(var_50, var_44);
        // geom_id, dist, normal, u, v, f, mesh_id = cast_ray(                                    <L 806>
        // geom_type,                                                                             <L 807>
        // geom_dataid,                                                                           <L 808>
        // geom_size,                                                                             <L 809>
        // flex_vertadr,                                                                          <L 810>
        // flex_edge,                                                                             <L 811>
        // flex_radius,                                                                           <L 812>
        // geom_xpos_in,                                                                          <L 813>
        // geom_xmat_in,                                                                          <L 814>
        // flexvert_xpos_in,                                                                      <L 815>
        // bvh_id,                                                                                <L 816>
        // group_root[worldid],                                                                   <L 817>
        var_53 = wp::address(var_group_root, var_0);
        // worldid,                                                                               <L 818>
        // bvh_ngeom,                                                                             <L 819>
        // bvh_nflexgeom,                                                                         <L 820>
        // enabled_geom_ids,                                                                      <L 821>
        // mesh_bvh_id,                                                                           <L 822>
        // hfield_bvh_id,                                                                         <L 823>
        // flex_geom_flexid,                                                                      <L 824>
        // flex_geom_edgeid,                                                                      <L 825>
        // flex_bvh_id,                                                                           <L 826>
        // flex_group_root,                                                                       <L 827>
        // ray_origin_world,                                                                      <L 828>
        // ray_dir_world,                                                                         <L 829>
        // float(MJ_MAXVAL),                                                                      <L 830>
        var_55 = wp::float(var_54);
        // wp.static(rc_static["enable_backface_culling"]),                                       <L 831>
        var_64 = wp::load(var_53);
        _make_cast_ray__locals__cast_ray_0(var_geom_type, var_geom_dataid, var_geom_size, var_flex_vertadr, var_flex_edge, var_flex_radius, var_geom_xpos_in, var_geom_xmat_in, var_flexvert_xpos_in, var_bvh_id, var_64, var_0, var_bvh_ngeom, var_bvh_nflexgeom, var_enabled_geom_ids, var_mesh_bvh_id, var_hfield_bvh_id, var_flex_geom_flexid, var_flex_geom_edgeid, var_flex_bvh_id, var_flex_group_root, var_47, var_52, var_55, var_56, var_57, var_58, var_59, var_60, var_61, var_62, var_63);
        // splat_color = wp.vec3(0.0)                                                             <L 834>
        var_66 = wp::vec_t<3, wp::float32>(var_65);
        // splat_transmittance = float(1.0)                                                       <L 835>
        var_68 = wp::float(var_67);
        // splat_depth = float(-1.0)                                                              <L 836>
        var_70 = wp::float(var_69);
        // if wp.static(has_splats):                                                              <L 837>
        // if render_seg[camid] and geom_id != -1:                                                <L 850>
        var_73 = wp::address(var_render_seg, var_3);
        var_74 = wp::load(var_73);
        var_72 = var_74;
        if (var_72) {
            var_76 = (var_57 != var_75);
            var_72 = var_72 && var_76;
        }
        if (var_72) {
            // if geom_id == -2:                                                                  <L 851>
            var_78 = (var_57 == var_77);
            if (var_78) {
                // seg_out[worldid, seg_adr[camid] + rayid_local] = wp.vec2i(mesh_id, int(ObjType.FLEX))       <L 852>
                var_81 = wp::int(var_80);
                var_82 = wp::vec_t<2, wp::int32>(var_63, var_81);
                var_83 = wp::address(var_seg_adr, var_3);
                var_85 = wp::load(var_83);
                var_84 = wp::add(var_85, var_5);
                wp::array_store(var_seg_out, var_0, var_84, var_82);
            }
            if (!var_78) {
                // seg_out[worldid, seg_adr[camid] + rayid_local] = wp.vec2i(geom_id, int(ObjType.GEOM))       <L 854>
                var_88 = wp::int(var_87);
                var_89 = wp::vec_t<2, wp::int32>(var_57, var_88);
                var_90 = wp::address(var_seg_adr, var_3);
                var_92 = wp::load(var_90);
                var_91 = wp::add(var_92, var_5);
                wp::array_store(var_seg_out, var_0, var_91, var_89);
            }
        }
        // if geom_id == -1:                                                                      <L 857>
        var_94 = (var_57 == var_93);
        if (var_94) {
            // if render_depth[camid]:                                                            <L 858>
            var_95 = wp::address(var_render_depth, var_3);
            var_96 = wp::load(var_95);
            if (var_96) {
                // depth = 0.0                                                                    <L 859>
                // if wp.static(has_splats):                                                      <L 860>
                // depth_out[worldid, depth_adr[camid] + rayid_local] = depth                     <L 863>
                var_99 = wp::address(var_depth_adr, var_3);
                var_101 = wp::load(var_99);
                var_100 = wp::add(var_101, var_5);
                wp::array_store(var_depth_out, var_0, var_100, var_97);
            }
            var_102 = wp::load(var_95);
            // if wp.static(rc_static["render_skybox"]) and render_rgb[camid]:                    <L 864>
            var_103 = var_104;
            if (var_103) {
                var_105 = wp::address(var_render_rgb, var_3);
                var_106 = wp::load(var_105);
                var_103 = var_103 && var_106;
            }
            if (var_103) {
                // skybox_id = skybox_tex_id[worldid % skybox_tex_id.shape[0]]                    <L 865>
                var_107 = &(var_skybox_tex_id.shape);
                var_110 = wp::load(var_107);
                var_109 = wp::extract(var_110, var_108);
                var_111 = wp::mod(var_0, var_109);
                var_112 = wp::address(var_skybox_tex_id, var_111);
                var_114 = wp::load(var_112);
                var_113 = wp::copy(var_114);
                // skybox_color = sample_skybox(                                                  <L 866>
                // textures[skybox_id],                                                           <L 867>
                var_115 = wp::address(var_textures, var_113);
                // 1.0 / float(skybox_face_width[worldid % skybox_face_width.shape[0]]),          <L 868>
                var_117 = &(var_skybox_face_width.shape);
                var_120 = wp::load(var_117);
                var_119 = wp::extract(var_120, var_118);
                var_121 = wp::mod(var_0, var_119);
                var_122 = wp::address(var_skybox_face_width, var_121);
                var_124 = wp::load(var_122);
                var_123 = wp::float(var_124);
                var_125 = wp::div(var_116, var_123);
                // ray_dir_world,                                                                 <L 869>
                var_127 = wp::load(var_115);
                var_126 = sample_skybox_0(var_127, var_125, var_52);
                // if wp.static(has_splats):                                                      <L 871>
                // rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32(          <L 873>
                // skybox_color[0] * 255.0,                                                       <L 874>
                var_130 = wp::extract(var_126, var_129);
                var_132 = wp::mul(var_130, var_131);
                // skybox_color[1] * 255.0,                                                       <L 875>
                var_134 = wp::extract(var_126, var_133);
                var_136 = wp::mul(var_134, var_135);
                // skybox_color[2] * 255.0,                                                       <L 876>
                var_138 = wp::extract(var_126, var_137);
                var_140 = wp::mul(var_138, var_139);
                // 255.0,                                                                         <L 877>
                var_142 = gl_rounding__locals__pack_rgba_round_0(var_132, var_136, var_140, var_141);
                // rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32(          <L 873>
                var_143 = wp::address(var_rgb_adr, var_3);
                var_145 = wp::load(var_143);
                var_144 = wp::add(var_145, var_5);
                wp::array_store(var_rgb_out, var_0, var_144, var_142);
            }
            if (!var_103) {
                // elif render_rgb[camid]:                                                        <L 879>
                var_146 = wp::address(var_render_rgb, var_3);
                var_147 = wp::load(var_146);
                if (var_147) {
                    // if wp.static(has_splats):                                                  <L 880>
                    // rgb_out[worldid, rgb_adr[camid] + rayid_local] = wp.static(rc_static["background_color"])       <L 892>
                    var_150 = wp::address(var_rgb_adr, var_3);
                    var_152 = wp::load(var_150);
                    var_151 = wp::add(var_152, var_5);
                    wp::array_store(var_rgb_out, var_0, var_151, var_149);
                }
                var_153 = wp::load(var_146);
            }
            // return                                                                             <L 893>
            return;
        }
        // if render_depth[camid]:                                                                <L 895>
        var_154 = wp::address(var_render_depth, var_3);
        var_155 = wp::load(var_154);
        if (var_155) {
            // depth = dist                                                                       <L 900>
            var_156 = wp::copy(var_58);
            // if wp.static(has_splats):                                                          <L 901>
            // depth_out[worldid, depth_adr[camid] + rayid_local] = depth * -ray_dir_local_cam[2]       <L 904>
            var_159 = wp::extract(var_44, var_158);
            var_160 = wp::neg(var_159);
            var_161 = wp::mul(var_156, var_160);
            var_162 = wp::address(var_depth_adr, var_3);
            var_164 = wp::load(var_162);
            var_163 = wp::add(var_164, var_5);
            wp::array_store(var_depth_out, var_0, var_163, var_161);
        }
        var_165 = wp::load(var_154);
        var_167 = wp::load(var_154);
        var_166 = wp::where(var_167, var_156, var_97);
        // if not render_rgb[camid]:                                                              <L 906>
        var_168 = wp::address(var_render_rgb, var_3);
        var_170 = wp::load(var_168);
        var_169 = wp::unot(var_170);
        if (var_169) {
            // return                                                                             <L 907>
            return;
        }
        // hit_point = ray_origin_world + ray_dir_world * dist                                    <L 910>
        var_171 = wp::mul(var_52, var_58);
        var_172 = wp::add(var_47, var_171);
        // if geom_id == -2:                                                                      <L 912>
        var_174 = (var_57 == var_173);
        if (var_174) {
            // color = flex_rgba[mesh_id]                                                         <L 914>
            var_175 = wp::address(var_flex_rgba, var_63);
            var_177 = wp::load(var_175);
            var_176 = wp::copy(var_177);
        }
        if (!var_174) {
            // elif geom_matid[worldid % geom_matid.shape[0], geom_id] == -1:                     <L 915>
            var_178 = &(var_geom_matid.shape);
            var_181 = wp::load(var_178);
            var_180 = wp::extract(var_181, var_179);
            var_182 = wp::mod(var_0, var_180);
            var_183 = wp::address(var_geom_matid, var_182, var_57);
            var_186 = wp::load(var_183);
            var_185 = (var_186 == var_184);
            if (var_185) {
                // color = geom_rgba[worldid % geom_rgba.shape[0], geom_id]                       <L 916>
                var_187 = &(var_geom_rgba.shape);
                var_190 = wp::load(var_187);
                var_189 = wp::extract(var_190, var_188);
                var_191 = wp::mod(var_0, var_189);
                var_192 = wp::address(var_geom_rgba, var_191, var_57);
                var_194 = wp::load(var_192);
                var_193 = wp::copy(var_194);
            }
            if (!var_185) {
                // color = mat_rgba[worldid % mat_rgba.shape[0], geom_matid[worldid % geom_matid.shape[0], geom_id]]       <L 918>
                var_195 = &(var_mat_rgba.shape);
                var_198 = wp::load(var_195);
                var_197 = wp::extract(var_198, var_196);
                var_199 = wp::mod(var_0, var_197);
                var_200 = &(var_geom_matid.shape);
                var_203 = wp::load(var_200);
                var_202 = wp::extract(var_203, var_201);
                var_204 = wp::mod(var_0, var_202);
                var_205 = wp::address(var_geom_matid, var_204, var_57);
                var_207 = wp::load(var_205);
                var_206 = wp::address(var_mat_rgba, var_199, var_207);
                var_209 = wp::load(var_206);
                var_208 = wp::copy(var_209);
            }
            var_210 = wp::where(var_185, var_193, var_208);
        }
        var_211 = wp::where(var_174, var_176, var_210);
        // base_color = wp.vec3(color[0], color[1], color[2])                                     <L 920>
        var_213 = wp::extract(var_211, var_212);
        var_215 = wp::extract(var_211, var_214);
        var_217 = wp::extract(var_211, var_216);
        var_218 = wp::vec_t<3, wp::float32>(var_213, var_215, var_217);
        // if wp.static(rc_static["use_textures"]):                                               <L 922>
        // if geom_id != -2:                                                                      <L 923>
        var_221 = (var_57 != var_220);
        if (var_221) {
            // mat_id = geom_matid[worldid % geom_matid.shape[0], geom_id]                        <L 924>
            var_222 = &(var_geom_matid.shape);
            var_225 = wp::load(var_222);
            var_224 = wp::extract(var_225, var_223);
            var_226 = wp::mod(var_0, var_224);
            var_227 = wp::address(var_geom_matid, var_226, var_57);
            var_229 = wp::load(var_227);
            var_228 = wp::copy(var_229);
            // if mat_id >= 0:                                                                    <L 925>
            var_231 = (var_228 >= var_230);
            if (var_231) {
                // tex_id = mat_texid[worldid % mat_texid.shape[0], mat_id, 1]                    <L 926>
                var_232 = &(var_mat_texid.shape);
                var_235 = wp::load(var_232);
                var_234 = wp::extract(var_235, var_233);
                var_236 = wp::mod(var_0, var_234);
                var_238 = wp::address(var_mat_texid, var_236, var_228, var_237);
                var_240 = wp::load(var_238);
                var_239 = wp::copy(var_240);
                // if tex_id >= 0:                                                                <L 927>
                var_242 = (var_239 >= var_241);
                if (var_242) {
                    // tex_color = sample_texture(                                                <L 928>
                    // geom_type,                                                                 <L 929>
                    // mesh_faceadr,                                                              <L 930>
                    // geom_id,                                                                   <L 931>
                    // mat_texrepeat[worldid % mat_texrepeat.shape[0], mat_id],                   <L 932>
                    var_243 = &(var_mat_texrepeat.shape);
                    var_246 = wp::load(var_243);
                    var_245 = wp::extract(var_246, var_244);
                    var_247 = wp::mod(var_0, var_245);
                    var_248 = wp::address(var_mat_texrepeat, var_247, var_228);
                    // textures[tex_id],                                                          <L 933>
                    var_249 = wp::address(var_textures, var_239);
                    // geom_xpos_in[worldid, geom_id],                                            <L 934>
                    var_250 = wp::address(var_geom_xpos_in, var_0, var_57);
                    // geom_xmat_in[worldid, geom_id],                                            <L 935>
                    var_251 = wp::address(var_geom_xmat_in, var_0, var_57);
                    // mesh_facetexcoord,                                                         <L 936>
                    // mesh_texcoord,                                                             <L 937>
                    // mesh_texcoord_offsets,                                                     <L 938>
                    // hit_point,                                                                 <L 939>
                    // u,                                                                         <L 940>
                    // v,                                                                         <L 941>
                    // f,                                                                         <L 942>
                    // mesh_id,                                                                   <L 943>
                    var_253 = wp::load(var_248);
                    var_254 = wp::load(var_249);
                    var_255 = wp::load(var_250);
                    var_256 = wp::load(var_251);
                    var_252 = sample_texture_0(var_geom_type, var_mesh_faceadr, var_57, var_253, var_254, var_255, var_256, var_mesh_facetexcoord, var_mesh_texcoord, var_mesh_texcoord_offsets, var_172, var_60, var_61, var_62, var_63);
                    // base_color = wp.cw_mul(base_color, tex_color)                              <L 945>
                    var_257 = wp::cw_mul(var_218, var_252);
                }
                var_258 = wp::where(var_242, var_257, var_218);
            }
            var_259 = wp::where(var_231, var_258, var_218);
        }
        var_260 = wp::where(var_221, var_259, var_218);
        // mat_spec = DEFAULT_MAT_SPECULAR                                                        <L 947>
        var_262 = wp::copy(var_261);
        // mat_shin_exp = DEFAULT_MAT_SHININESS_EXPONENT                                          <L 948>
        var_264 = wp::copy(var_263);
        // mat_emis = DEFAULT_MAT_EMISSION                                                        <L 949>
        var_266 = wp::copy(var_265);
        // if wp.static(rc_static["enable_specular_or_emission"]):                                <L 950>
        // if geom_id != -2:                                                                      <L 951>
        var_269 = (var_57 != var_268);
        if (var_269) {
            // mat_id_for_spec = geom_matid[worldid % geom_matid.shape[0], geom_id]               <L 952>
            var_270 = &(var_geom_matid.shape);
            var_273 = wp::load(var_270);
            var_272 = wp::extract(var_273, var_271);
            var_274 = wp::mod(var_0, var_272);
            var_275 = wp::address(var_geom_matid, var_274, var_57);
            var_277 = wp::load(var_275);
            var_276 = wp::copy(var_277);
            // if mat_id_for_spec >= 0:                                                           <L 953>
            var_279 = (var_276 >= var_278);
            if (var_279) {
                // if wp.static(rc_static["enable_specular"]):                                    <L 954>
                // mat_spec = mat_specular[worldid % mat_specular.shape[0], mat_id_for_spec]       <L 955>
                var_281 = &(var_mat_specular.shape);
                var_284 = wp::load(var_281);
                var_283 = wp::extract(var_284, var_282);
                var_285 = wp::mod(var_0, var_283);
                var_286 = wp::address(var_mat_specular, var_285, var_276);
                var_288 = wp::load(var_286);
                var_287 = wp::copy(var_288);
                // mat_shin_exp = mat_shininess[worldid % mat_shininess.shape[0], mat_id_for_spec] * MAX_SHININESS       <L 956>
                var_289 = &(var_mat_shininess.shape);
                var_292 = wp::load(var_289);
                var_291 = wp::extract(var_292, var_290);
                var_293 = wp::mod(var_0, var_291);
                var_294 = wp::address(var_mat_shininess, var_293, var_276);
                var_297 = wp::load(var_294);
                var_296 = wp::mul(var_297, var_295);
                // if wp.static(rc_static["enable_emission"]):                                    <L 957>
                // mat_emis = mat_emission[worldid % mat_emission.shape[0], mat_id_for_spec]       <L 958>
                var_299 = &(var_mat_emission.shape);
                var_302 = wp::load(var_299);
                var_301 = wp::extract(var_302, var_300);
                var_303 = wp::mod(var_0, var_301);
                var_304 = wp::address(var_mat_emission, var_303, var_276);
                var_306 = wp::load(var_304);
                var_305 = wp::copy(var_306);
            }
            var_307 = wp::where(var_279, var_287, var_262);
            var_308 = wp::where(var_279, var_296, var_264);
            var_309 = wp::where(var_279, var_305, var_266);
        }
        var_310 = wp::where(var_269, var_307, var_262);
        var_311 = wp::where(var_269, var_308, var_264);
        var_312 = wp::where(var_269, var_309, var_266);
        // result = wp.vec3(0.0)                                                                  <L 960>
        var_314 = wp::vec_t<3, wp::float32>(var_313);
        // if wp.static(rc_static["enable_emission"]):                                            <L 961>
        // result = base_color * mat_emis                                                         <L 962>
        var_316 = wp::mul(var_260, var_312);
        // if wp.static(rc_static["use_ambient_lighting"]):                                       <L 964>
        // if wp.static(rc_static["headlight_active"]):                                           <L 965>
        // result = result + wp.cw_mul(base_color, wp.static(rc_static["headlight_ambient"]))       <L 966>
        var_320 = wp::cw_mul(var_260, var_319);
        var_321 = wp::add(var_316, var_320);
        // if wp.static(rc_static["enable_per_light_ambient"]):                                   <L 969>
        // for light_index in range(wp.static(M_NLIGHT)):                                         <L 970>
        // if light_active[worldid % light_active.shape[0], light_index]:                         <L 971>
        var_324 = &(var_light_active.shape);
        var_327 = wp::load(var_324);
        var_326 = wp::extract(var_327, var_325);
        var_328 = wp::mod(var_0, var_326);
        var_329 = wp::address(var_light_active, var_328, var_323);
        var_330 = wp::load(var_329);
        if (var_330) {
            // result = result + wp.cw_mul(base_color, light_ambient[worldid % light_ambient.shape[0], light_index])       <L 972>
            var_331 = &(var_light_ambient.shape);
            var_334 = wp::load(var_331);
            var_333 = wp::extract(var_334, var_332);
            var_335 = wp::mod(var_0, var_333);
            var_336 = wp::address(var_light_ambient, var_335, var_323);
            var_338 = wp::load(var_336);
            var_337 = wp::cw_mul(var_260, var_338);
            var_339 = wp::add(var_321, var_337);
        }
        var_340 = wp::load(var_329);
        var_342 = wp::load(var_329);
        var_341 = wp::where(var_342, var_339, var_321);
        // if light_active[worldid % light_active.shape[0], light_index]:                         <L 971>
        var_344 = &(var_light_active.shape);
        var_347 = wp::load(var_344);
        var_346 = wp::extract(var_347, var_345);
        var_348 = wp::mod(var_0, var_346);
        var_349 = wp::address(var_light_active, var_348, var_343);
        var_350 = wp::load(var_349);
        if (var_350) {
            // result = result + wp.cw_mul(base_color, light_ambient[worldid % light_ambient.shape[0], light_index])       <L 972>
            var_351 = &(var_light_ambient.shape);
            var_354 = wp::load(var_351);
            var_353 = wp::extract(var_354, var_352);
            var_355 = wp::mod(var_0, var_353);
            var_356 = wp::address(var_light_ambient, var_355, var_343);
            var_358 = wp::load(var_356);
            var_357 = wp::cw_mul(var_260, var_358);
            var_359 = wp::add(var_341, var_357);
        }
        var_360 = wp::load(var_349);
        var_362 = wp::load(var_349);
        var_361 = wp::where(var_362, var_359, var_341);
        // view_dir = -ray_dir_world                                                              <L 974>
        var_363 = wp::neg(var_52);
        // light_cutoff_worldid = light_cutoff[worldid % light_cutoff.shape[0]]                   <L 976>
        var_364 = &(var_light_cutoff.shape);
        var_367 = wp::load(var_364);
        var_366 = wp::extract(var_367, var_365);
        var_368 = wp::mod(var_0, var_366);
        var_369 = wp::view(var_light_cutoff, var_368);
        // light_active_worldid = light_active[worldid % light_active.shape[0]]                   <L 977>
        var_370 = &(var_light_active.shape);
        var_373 = wp::load(var_370);
        var_372 = wp::extract(var_373, var_371);
        var_374 = wp::mod(var_0, var_372);
        var_375 = wp::view(var_light_active, var_374);
        // light_type_worldid = light_type[worldid % light_type.shape[0]]                         <L 978>
        var_376 = &(var_light_type.shape);
        var_379 = wp::load(var_376);
        var_378 = wp::extract(var_379, var_377);
        var_380 = wp::mod(var_0, var_378);
        var_381 = wp::view(var_light_type, var_380);
        // light_castshadow_worldid = light_castshadow[worldid % light_castshadow.shape[0]]       <L 979>
        var_382 = &(var_light_castshadow.shape);
        var_385 = wp::load(var_382);
        var_384 = wp::extract(var_385, var_383);
        var_386 = wp::mod(var_0, var_384);
        var_387 = wp::view(var_light_castshadow, var_386);
        // light_xpos_in_worldid = light_xpos_in[worldid]                                         <L 980>
        var_388 = wp::view(var_light_xpos_in, var_0);
        // light_xdir_in_worldid = light_xdir_in[worldid]                                         <L 981>
        var_389 = wp::view(var_light_xdir_in, var_0);
        // light_attenuation_worldid = light_attenuation[worldid % light_attenuation.shape[0]]       <L 982>
        var_390 = &(var_light_attenuation.shape);
        var_393 = wp::load(var_390);
        var_392 = wp::extract(var_393, var_391);
        var_394 = wp::mod(var_0, var_392);
        var_395 = wp::view(var_light_attenuation, var_394);
        // light_exponent_worldid = light_exponent[worldid % light_exponent.shape[0]]             <L 983>
        var_396 = &(var_light_exponent.shape);
        var_399 = wp::load(var_396);
        var_398 = wp::extract(var_399, var_397);
        var_400 = wp::mod(var_0, var_398);
        var_401 = wp::view(var_light_exponent, var_400);
        // light_diffuse_worldid = light_diffuse[worldid % light_diffuse.shape[0]]                <L 984>
        var_402 = &(var_light_diffuse.shape);
        var_405 = wp::load(var_402);
        var_404 = wp::extract(var_405, var_403);
        var_406 = wp::mod(var_0, var_404);
        var_407 = wp::view(var_light_diffuse, var_406);
        // light_specular_worldid = light_specular[worldid % light_specular.shape[0]]             <L 985>
        var_408 = &(var_light_specular.shape);
        var_411 = wp::load(var_408);
        var_410 = wp::extract(var_411, var_409);
        var_412 = wp::mod(var_0, var_410);
        var_413 = wp::view(var_light_specular, var_412);
        // for light_index in range(wp.static(M_NLIGHT)):                                         <L 987>
        // diff_rgb, spec_rgb = compute_lighting(                                                 <L 988>
        // geom_type,                                                                             <L 989>
        // geom_dataid,                                                                           <L 990>
        // geom_size,                                                                             <L 991>
        // flex_vertadr,                                                                          <L 992>
        // flex_edge,                                                                             <L 993>
        // flex_radius,                                                                           <L 994>
        // geom_xpos_in,                                                                          <L 995>
        // geom_xmat_in,                                                                          <L 996>
        // flexvert_xpos_in,                                                                      <L 997>
        // use_shadows,                                                                           <L 998>
        // bvh_id,                                                                                <L 999>
        // group_root[worldid],                                                                   <L 1000>
        var_415 = wp::address(var_group_root, var_0);
        // bvh_ngeom,                                                                             <L 1001>
        // bvh_nflexgeom,                                                                         <L 1002>
        // enabled_geom_ids,                                                                      <L 1003>
        // worldid,                                                                               <L 1004>
        // mesh_bvh_id,                                                                           <L 1005>
        // hfield_bvh_id,                                                                         <L 1006>
        // flex_geom_flexid,                                                                      <L 1007>
        // flex_geom_edgeid,                                                                      <L 1008>
        // flex_bvh_id,                                                                           <L 1009>
        // flex_group_root,                                                                       <L 1010>
        // light_active_worldid[light_index],                                                     <L 1011>
        var_416 = wp::address(var_375, var_414);
        // light_type_worldid[light_index],                                                       <L 1012>
        var_417 = wp::address(var_381, var_414);
        // light_castshadow_worldid[light_index],                                                 <L 1013>
        var_418 = wp::address(var_387, var_414);
        // light_xpos_in_worldid[light_index],                                                    <L 1014>
        var_419 = wp::address(var_388, var_414);
        // light_xdir_in_worldid[light_index],                                                    <L 1015>
        var_420 = wp::address(var_389, var_414);
        // light_attenuation_worldid[light_index],                                                <L 1016>
        var_421 = wp::address(var_395, var_414);
        // light_cutoff_worldid[light_index] * wp.static(wp.pi / 180.0),                          <L 1017>
        var_422 = wp::address(var_369, var_414);
        var_425 = wp::load(var_422);
        var_424 = wp::mul(var_425, var_423);
        // light_exponent_worldid[light_index],                                                   <L 1018>
        var_426 = wp::address(var_401, var_414);
        // light_diffuse_worldid[light_index],                                                    <L 1019>
        var_427 = wp::address(var_407, var_414);
        // light_specular_worldid[light_index],                                                   <L 1020>
        var_428 = wp::address(var_413, var_414);
        // normal,                                                                                <L 1021>
        // hit_point,                                                                             <L 1022>
        // view_dir,                                                                              <L 1023>
        // mat_spec,                                                                              <L 1024>
        // mat_shin_exp,                                                                          <L 1025>
        // wp.static(rc_static["enable_backface_culling"]),                                       <L 1026>
        // wp.static(rc_static["enable_specular"]),                                               <L 1027>
        // wp.static(rc_static["light_attenuation_is_default"]),                                  <L 1028>
        // wp.static(rc_static["has_spot_lights"]),                                               <L 1029>
        var_435 = wp::load(var_415);
        var_436 = wp::load(var_416);
        var_437 = wp::load(var_417);
        var_438 = wp::load(var_418);
        var_439 = wp::load(var_419);
        var_440 = wp::load(var_420);
        var_441 = wp::load(var_421);
        var_442 = wp::load(var_426);
        var_443 = wp::load(var_427);
        var_444 = wp::load(var_428);
        _make_compute_lighting__locals__compute_lighting_0(var_geom_type, var_geom_dataid, var_geom_size, var_flex_vertadr, var_flex_edge, var_flex_radius, var_geom_xpos_in, var_geom_xmat_in, var_flexvert_xpos_in, var_use_shadows, var_bvh_id, var_435, var_bvh_ngeom, var_bvh_nflexgeom, var_enabled_geom_ids, var_0, var_mesh_bvh_id, var_hfield_bvh_id, var_flex_geom_flexid, var_flex_geom_edgeid, var_flex_bvh_id, var_flex_group_root, var_436, var_437, var_438, var_439, var_440, var_441, var_424, var_442, var_443, var_444, var_59, var_172, var_363, var_310, var_311, var_429, var_430, var_431, var_432, var_433, var_434);
        // result = result + wp.cw_mul(base_color, diff_rgb) + spec_rgb                           <L 1031>
        var_445 = wp::cw_mul(var_260, var_433);
        var_446 = wp::add(var_361, var_445);
        var_447 = wp::add(var_446, var_434);
        // diff_rgb, spec_rgb = compute_lighting(                                                 <L 988>
        // geom_type,                                                                             <L 989>
        // geom_dataid,                                                                           <L 990>
        // geom_size,                                                                             <L 991>
        // flex_vertadr,                                                                          <L 992>
        // flex_edge,                                                                             <L 993>
        // flex_radius,                                                                           <L 994>
        // geom_xpos_in,                                                                          <L 995>
        // geom_xmat_in,                                                                          <L 996>
        // flexvert_xpos_in,                                                                      <L 997>
        // use_shadows,                                                                           <L 998>
        // bvh_id,                                                                                <L 999>
        // group_root[worldid],                                                                   <L 1000>
        var_449 = wp::address(var_group_root, var_0);
        // bvh_ngeom,                                                                             <L 1001>
        // bvh_nflexgeom,                                                                         <L 1002>
        // enabled_geom_ids,                                                                      <L 1003>
        // worldid,                                                                               <L 1004>
        // mesh_bvh_id,                                                                           <L 1005>
        // hfield_bvh_id,                                                                         <L 1006>
        // flex_geom_flexid,                                                                      <L 1007>
        // flex_geom_edgeid,                                                                      <L 1008>
        // flex_bvh_id,                                                                           <L 1009>
        // flex_group_root,                                                                       <L 1010>
        // light_active_worldid[light_index],                                                     <L 1011>
        var_450 = wp::address(var_375, var_448);
        // light_type_worldid[light_index],                                                       <L 1012>
        var_451 = wp::address(var_381, var_448);
        // light_castshadow_worldid[light_index],                                                 <L 1013>
        var_452 = wp::address(var_387, var_448);
        // light_xpos_in_worldid[light_index],                                                    <L 1014>
        var_453 = wp::address(var_388, var_448);
        // light_xdir_in_worldid[light_index],                                                    <L 1015>
        var_454 = wp::address(var_389, var_448);
        // light_attenuation_worldid[light_index],                                                <L 1016>
        var_455 = wp::address(var_395, var_448);
        // light_cutoff_worldid[light_index] * wp.static(wp.pi / 180.0),                          <L 1017>
        var_456 = wp::address(var_369, var_448);
        var_459 = wp::load(var_456);
        var_458 = wp::mul(var_459, var_457);
        // light_exponent_worldid[light_index],                                                   <L 1018>
        var_460 = wp::address(var_401, var_448);
        // light_diffuse_worldid[light_index],                                                    <L 1019>
        var_461 = wp::address(var_407, var_448);
        // light_specular_worldid[light_index],                                                   <L 1020>
        var_462 = wp::address(var_413, var_448);
        // normal,                                                                                <L 1021>
        // hit_point,                                                                             <L 1022>
        // view_dir,                                                                              <L 1023>
        // mat_spec,                                                                              <L 1024>
        // mat_shin_exp,                                                                          <L 1025>
        // wp.static(rc_static["enable_backface_culling"]),                                       <L 1026>
        // wp.static(rc_static["enable_specular"]),                                               <L 1027>
        // wp.static(rc_static["light_attenuation_is_default"]),                                  <L 1028>
        // wp.static(rc_static["has_spot_lights"]),                                               <L 1029>
        var_469 = wp::load(var_449);
        var_470 = wp::load(var_450);
        var_471 = wp::load(var_451);
        var_472 = wp::load(var_452);
        var_473 = wp::load(var_453);
        var_474 = wp::load(var_454);
        var_475 = wp::load(var_455);
        var_476 = wp::load(var_460);
        var_477 = wp::load(var_461);
        var_478 = wp::load(var_462);
        _make_compute_lighting__locals__compute_lighting_0(var_geom_type, var_geom_dataid, var_geom_size, var_flex_vertadr, var_flex_edge, var_flex_radius, var_geom_xpos_in, var_geom_xmat_in, var_flexvert_xpos_in, var_use_shadows, var_bvh_id, var_469, var_bvh_ngeom, var_bvh_nflexgeom, var_enabled_geom_ids, var_0, var_mesh_bvh_id, var_hfield_bvh_id, var_flex_geom_flexid, var_flex_geom_edgeid, var_flex_bvh_id, var_flex_group_root, var_470, var_471, var_472, var_473, var_474, var_475, var_458, var_476, var_477, var_478, var_59, var_172, var_363, var_310, var_311, var_463, var_464, var_465, var_466, var_467, var_468);
        // result = result + wp.cw_mul(base_color, diff_rgb) + spec_rgb                           <L 1031>
        var_479 = wp::cw_mul(var_260, var_467);
        var_480 = wp::add(var_447, var_479);
        var_481 = wp::add(var_480, var_468);
        // if wp.static(rc_static["headlight_active"]):                                           <L 1034>
        // cam_pos = ray_origin_world                                                             <L 1035>
        var_483 = wp::copy(var_47);
        // cam_fwd = -cam_mat_world[:, 2]                                                         <L 1036>
        var_484 = wp::slice_t(var_485, var_486, var_487);
        var_489 = wp::extract<3>(var_50, var_484, var_488);
        var_490 = wp::neg(var_489);
        // hl_diff, hl_spec = compute_lighting(                                                   <L 1037>
        // geom_type,                                                                             <L 1038>
        // geom_dataid,                                                                           <L 1039>
        // geom_size,                                                                             <L 1040>
        // flex_vertadr,                                                                          <L 1041>
        // flex_edge,                                                                             <L 1042>
        // flex_radius,                                                                           <L 1043>
        // geom_xpos_in,                                                                          <L 1044>
        // geom_xmat_in,                                                                          <L 1045>
        // flexvert_xpos_in,                                                                      <L 1046>
        // use_shadows,                                                                           <L 1047>
        // bvh_id,                                                                                <L 1048>
        // group_root[worldid],                                                                   <L 1049>
        var_491 = wp::address(var_group_root, var_0);
        // bvh_ngeom,                                                                             <L 1050>
        // bvh_nflexgeom,                                                                         <L 1051>
        // enabled_geom_ids,                                                                      <L 1052>
        // worldid,                                                                               <L 1053>
        // mesh_bvh_id,                                                                           <L 1054>
        // hfield_bvh_id,                                                                         <L 1055>
        // flex_geom_flexid,                                                                      <L 1056>
        // flex_geom_edgeid,                                                                      <L 1057>
        // flex_bvh_id,                                                                           <L 1058>
        // flex_group_root,                                                                       <L 1059>
        // True,                                                                                  <L 1060>
        // 1,                                                                                     <L 1061>
        // False,                                                                                 <L 1062>
        // cam_pos,                                                                               <L 1063>
        // cam_fwd,                                                                               <L 1064>
        // wp.vec3(1.0, 0.0, 0.0),                                                                <L 1065>
        var_498 = wp::vec_t<3, wp::float32>(var_495, var_496, var_497);
        // 0.0,                                                                                   <L 1066>
        // 0.0,                                                                                   <L 1067>
        // wp.static(rc_static["headlight_diffuse"]),                                             <L 1068>
        // wp.static(rc_static["headlight_specular"]),                                            <L 1069>
        // normal,                                                                                <L 1070>
        // hit_point,                                                                             <L 1071>
        // view_dir,                                                                              <L 1072>
        // mat_spec,                                                                              <L 1073>
        // mat_shin_exp,                                                                          <L 1074>
        // wp.static(rc_static["enable_backface_culling"]),                                       <L 1075>
        // wp.static(rc_static["enable_specular"]),                                               <L 1076>
        // True,                                                                                  <L 1077>
        // False,                                                                                 <L 1078>
        var_509 = wp::load(var_491);
        _make_compute_lighting__locals__compute_lighting_0(var_geom_type, var_geom_dataid, var_geom_size, var_flex_vertadr, var_flex_edge, var_flex_radius, var_geom_xpos_in, var_geom_xmat_in, var_flexvert_xpos_in, var_use_shadows, var_bvh_id, var_509, var_bvh_ngeom, var_bvh_nflexgeom, var_enabled_geom_ids, var_0, var_mesh_bvh_id, var_hfield_bvh_id, var_flex_geom_flexid, var_flex_geom_edgeid, var_flex_bvh_id, var_flex_group_root, var_492, var_493, var_494, var_483, var_490, var_498, var_499, var_500, var_501, var_502, var_59, var_172, var_363, var_310, var_311, var_503, var_504, var_505, var_506, var_507, var_508);
        // result = result + wp.cw_mul(base_color, hl_diff) + hl_spec                             <L 1080>
        var_510 = wp::cw_mul(var_260, var_507);
        var_511 = wp::add(var_481, var_510);
        var_512 = wp::add(var_511, var_508);
        // hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))                                     <L 1082>
        var_516 = wp::vec_t<3, wp::float32>(var_513, var_514, var_515);
        var_517 = wp::min(var_512, var_516);
        // hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))                                  <L 1083>
        var_521 = wp::vec_t<3, wp::float32>(var_518, var_519, var_520);
        var_522 = wp::max(var_517, var_521);
        // if wp.static(has_splats):                                                              <L 1084>
        // rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32(                  <L 1087>
        // hit_color[0] * 255.0,                                                                  <L 1088>
        var_525 = wp::extract(var_522, var_524);
        var_527 = wp::mul(var_525, var_526);
        // hit_color[1] * 255.0,                                                                  <L 1089>
        var_529 = wp::extract(var_522, var_528);
        var_531 = wp::mul(var_529, var_530);
        // hit_color[2] * 255.0,                                                                  <L 1090>
        var_533 = wp::extract(var_522, var_532);
        var_535 = wp::mul(var_533, var_534);
        // 255.0,                                                                                 <L 1091>
        var_537 = gl_rounding__locals__pack_rgba_round_0(var_527, var_531, var_535, var_536);
        // rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32(                  <L 1087>
        var_538 = wp::address(var_rgb_adr, var_3);
        var_540 = wp::load(var_538);
        var_539 = wp::add(var_540, var_5);
        wp::array_store(var_rgb_out, var_0, var_539, var_537);
}

