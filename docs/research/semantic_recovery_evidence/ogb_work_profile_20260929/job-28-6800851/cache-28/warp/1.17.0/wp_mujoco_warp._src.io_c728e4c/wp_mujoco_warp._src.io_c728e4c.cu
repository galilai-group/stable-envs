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


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render_util.py:66
static CUDA_CALLABLE wp::vec_t<3, wp::float32> compute_ray_0(
    wp::int32 var_projection,
    wp::float32 var_fovy,
    wp::vec_t<2, wp::float32> var_sensorsize,
    wp::vec_t<4, wp::float32> var_intrinsic,
    wp::int32 var_img_w,
    wp::int32 var_img_h,
    wp::int32 var_px,
    wp::int32 var_py,
    wp::float32 var_znear)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = -1.0;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    const wp::int32 var_15 = 1;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    bool var_27;
    wp::float32 var_28;
    bool var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.5;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::float32 var_41 = 0.5;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.5;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::float32 var_50 = 0.5;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 0.017453292519943295;
    wp::float32 var_55;
    const wp::float32 var_56 = 0.5;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.5;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.5;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    //---------
    // forward
    // def compute_ray(                                                                       <L 67>
    // if projection == ProjectionType.ORTHOGRAPHIC:                                          <L 84>
    var_1 = (var_projection == var_0);
    if (var_1) {
        // return wp.vec3(0.0, 0.0, -1.0)                                                     <L 85>
        var_5 = wp::vec_t<3, wp::float32>(var_2, var_3, var_4);
        return var_5;
    }
    // aspect = float(img_w) / float(img_h)                                                   <L 87>
    var_6 = wp::float(var_img_w);
    var_7 = wp::float(var_img_h);
    var_8 = wp::div(var_6, var_7);
    // sensor_h = sensorsize[1]                                                               <L 88>
    var_10 = wp::extract(var_sensorsize, var_9);
    // if sensor_h != 0.0:                                                                    <L 91>
    var_12 = (var_10 != var_11);
    if (var_12) {
        // fx = intrinsic[0]                                                                  <L 92>
        var_14 = wp::extract(var_intrinsic, var_13);
        // fy = intrinsic[1]                                                                  <L 93>
        var_16 = wp::extract(var_intrinsic, var_15);
        // cx = intrinsic[2]                                                                  <L 94>
        var_18 = wp::extract(var_intrinsic, var_17);
        // cy = intrinsic[3]                                                                  <L 95>
        var_20 = wp::extract(var_intrinsic, var_19);
        // sensor_w = sensorsize[0]                                                           <L 96>
        var_22 = wp::extract(var_sensorsize, var_21);
        // target_aspect = float(img_w) / float(img_h)                                        <L 98>
        var_23 = wp::float(var_img_w);
        var_24 = wp::float(var_img_h);
        var_25 = wp::div(var_23, var_24);
        // sensor_aspect = sensor_w / sensor_h                                                <L 99>
        var_26 = wp::div(var_22, var_10);
        // if target_aspect > sensor_aspect:                                                  <L 100>
        var_27 = (var_25 > var_26);
        if (var_27) {
            // sensor_h = sensor_w / target_aspect                                            <L 101>
            var_28 = wp::div(var_22, var_25);
        }
        if (!var_27) {
            // elif target_aspect < sensor_aspect:                                            <L 102>
            var_29 = (var_25 < var_26);
            if (var_29) {
                // sensor_w = sensor_h * target_aspect                                        <L 103>
                var_30 = wp::mul(var_10, var_25);
            }
            var_31 = wp::where(var_29, var_30, var_22);
        }
        var_32 = wp::where(var_27, var_28, var_10);
        var_33 = wp::where(var_27, var_22, var_31);
        // inv_fx_znear = znear / fx                                                          <L 105>
        var_34 = wp::div(var_znear, var_14);
        // inv_fy_znear = znear / fy                                                          <L 106>
        var_35 = wp::div(var_znear, var_16);
        // left = -inv_fx_znear * (sensor_w * 0.5 - cx)                                       <L 107>
        var_36 = wp::neg(var_34);
        var_38 = wp::mul(var_33, var_37);
        var_39 = wp::sub(var_38, var_18);
        var_40 = wp::mul(var_36, var_39);
        // right = inv_fx_znear * (sensor_w * 0.5 + cx)                                       <L 108>
        var_42 = wp::mul(var_33, var_41);
        var_43 = wp::add(var_42, var_18);
        var_44 = wp::mul(var_34, var_43);
        // top = inv_fy_znear * (sensor_h * 0.5 - cy)                                         <L 109>
        var_46 = wp::mul(var_32, var_45);
        var_47 = wp::sub(var_46, var_20);
        var_48 = wp::mul(var_35, var_47);
        // bottom = -inv_fy_znear * (sensor_h * 0.5 + cy)                                     <L 110>
        var_49 = wp::neg(var_35);
        var_51 = wp::mul(var_32, var_50);
        var_52 = wp::add(var_51, var_20);
        var_53 = wp::mul(var_49, var_52);
    }
    if (!var_12) {
        // fovy_rad = fovy * wp.static(wp.pi / 180.0)                                         <L 112>
        var_55 = wp::mul(var_fovy, var_54);
        // half_height = znear * wp.tan(0.5 * fovy_rad)                                       <L 113>
        var_57 = wp::mul(var_56, var_55);
        var_58 = wp::tan(var_57);
        var_59 = wp::mul(var_znear, var_58);
        // half_width = half_height * aspect                                                  <L 114>
        var_60 = wp::mul(var_59, var_8);
        // left = -half_width                                                                 <L 115>
        var_61 = wp::neg(var_60);
        // right = half_width                                                                 <L 116>
        var_62 = wp::copy(var_60);
        // top = half_height                                                                  <L 117>
        var_63 = wp::copy(var_59);
        // bottom = -half_height                                                              <L 118>
        var_64 = wp::neg(var_59);
    }
    var_65 = wp::where(var_12, var_32, var_10);
    var_66 = wp::where(var_12, var_40, var_61);
    var_67 = wp::where(var_12, var_44, var_62);
    var_68 = wp::where(var_12, var_48, var_63);
    var_69 = wp::where(var_12, var_53, var_64);
    // u = (float(px) + 0.5) / float(img_w)                                                   <L 120>
    var_70 = wp::float(var_px);
    var_72 = wp::add(var_70, var_71);
    var_73 = wp::float(var_img_w);
    var_74 = wp::div(var_72, var_73);
    // v = (float(py) + 0.5) / float(img_h)                                                   <L 121>
    var_75 = wp::float(var_py);
    var_77 = wp::add(var_75, var_76);
    var_78 = wp::float(var_img_h);
    var_79 = wp::div(var_77, var_78);
    // x = left + (right - left) * u                                                          <L 122>
    var_80 = wp::sub(var_67, var_66);
    var_81 = wp::mul(var_80, var_74);
    var_82 = wp::add(var_66, var_81);
    // y = top + (bottom - top) * v                                                           <L 123>
    var_83 = wp::sub(var_69, var_68);
    var_84 = wp::mul(var_83, var_79);
    var_85 = wp::add(var_68, var_84);
    // return wp.normalize(wp.vec3(x, y, -znear))                                             <L 125>
    var_86 = wp::neg(var_znear);
    var_87 = wp::vec_t<3, wp::float32>(var_82, var_85, var_86);
    var_88 = wp::normalize(var_87);
    return var_88;
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


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/render_util.py:66
static CUDA_CALLABLE void adj_compute_ray_0(
    wp::int32 var_projection,
    wp::float32 var_fovy,
    wp::vec_t<2, wp::float32> var_sensorsize,
    wp::vec_t<4, wp::float32> var_intrinsic,
    wp::int32 var_img_w,
    wp::int32 var_img_h,
    wp::int32 var_px,
    wp::int32 var_py,
    wp::float32 var_znear,
    wp::int32 & adj_projection,
    wp::float32 & adj_fovy,
    wp::vec_t<2, wp::float32> & adj_sensorsize,
    wp::vec_t<4, wp::float32> & adj_intrinsic,
    wp::int32 & adj_img_w,
    wp::int32 & adj_img_h,
    wp::int32 & adj_px,
    wp::int32 & adj_py,
    wp::float32 & adj_znear,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = -1.0;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    const wp::int32 var_15 = 1;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    bool var_27;
    wp::float32 var_28;
    bool var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.5;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::float32 var_41 = 0.5;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.5;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::float32 var_50 = 0.5;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 0.017453292519943295;
    wp::float32 var_55;
    const wp::float32 var_56 = 0.5;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.5;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.5;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    bool adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    bool adj_27 = {};
    wp::float32 adj_28 = {};
    bool adj_29 = {};
    wp::float32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::float32 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::float32 adj_70 = {};
    wp::float32 adj_71 = {};
    wp::float32 adj_72 = {};
    wp::float32 adj_73 = {};
    wp::float32 adj_74 = {};
    wp::float32 adj_75 = {};
    wp::float32 adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::float32 adj_80 = {};
    wp::float32 adj_81 = {};
    wp::float32 adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::float32 adj_86 = {};
    wp::vec_t<3, wp::float32> adj_87 = {};
    wp::vec_t<3, wp::float32> adj_88 = {};
    //---------
    // forward
    // def compute_ray(                                                                       <L 67>
    // if projection == ProjectionType.ORTHOGRAPHIC:                                          <L 84>
    var_1 = (var_projection == var_0);
    if (var_1) {
        // return wp.vec3(0.0, 0.0, -1.0)                                                     <L 85>
        var_5 = wp::vec_t<3, wp::float32>(var_2, var_3, var_4);
        goto label0;
    }
    // aspect = float(img_w) / float(img_h)                                                   <L 87>
    var_6 = wp::float(var_img_w);
    var_7 = wp::float(var_img_h);
    var_8 = wp::div(var_6, var_7);
    // sensor_h = sensorsize[1]                                                               <L 88>
    var_10 = wp::extract(var_sensorsize, var_9);
    // if sensor_h != 0.0:                                                                    <L 91>
    var_12 = (var_10 != var_11);
    if (var_12) {
        // fx = intrinsic[0]                                                                  <L 92>
        var_14 = wp::extract(var_intrinsic, var_13);
        // fy = intrinsic[1]                                                                  <L 93>
        var_16 = wp::extract(var_intrinsic, var_15);
        // cx = intrinsic[2]                                                                  <L 94>
        var_18 = wp::extract(var_intrinsic, var_17);
        // cy = intrinsic[3]                                                                  <L 95>
        var_20 = wp::extract(var_intrinsic, var_19);
        // sensor_w = sensorsize[0]                                                           <L 96>
        var_22 = wp::extract(var_sensorsize, var_21);
        // target_aspect = float(img_w) / float(img_h)                                        <L 98>
        var_23 = wp::float(var_img_w);
        var_24 = wp::float(var_img_h);
        var_25 = wp::div(var_23, var_24);
        // sensor_aspect = sensor_w / sensor_h                                                <L 99>
        var_26 = wp::div(var_22, var_10);
        // if target_aspect > sensor_aspect:                                                  <L 100>
        var_27 = (var_25 > var_26);
        if (var_27) {
            // sensor_h = sensor_w / target_aspect                                            <L 101>
            var_28 = wp::div(var_22, var_25);
        }
        if (!var_27) {
            // elif target_aspect < sensor_aspect:                                            <L 102>
            var_29 = (var_25 < var_26);
            if (var_29) {
                // sensor_w = sensor_h * target_aspect                                        <L 103>
                var_30 = wp::mul(var_10, var_25);
            }
            var_31 = wp::where(var_29, var_30, var_22);
        }
        var_32 = wp::where(var_27, var_28, var_10);
        var_33 = wp::where(var_27, var_22, var_31);
        // inv_fx_znear = znear / fx                                                          <L 105>
        var_34 = wp::div(var_znear, var_14);
        // inv_fy_znear = znear / fy                                                          <L 106>
        var_35 = wp::div(var_znear, var_16);
        // left = -inv_fx_znear * (sensor_w * 0.5 - cx)                                       <L 107>
        var_36 = wp::neg(var_34);
        var_38 = wp::mul(var_33, var_37);
        var_39 = wp::sub(var_38, var_18);
        var_40 = wp::mul(var_36, var_39);
        // right = inv_fx_znear * (sensor_w * 0.5 + cx)                                       <L 108>
        var_42 = wp::mul(var_33, var_41);
        var_43 = wp::add(var_42, var_18);
        var_44 = wp::mul(var_34, var_43);
        // top = inv_fy_znear * (sensor_h * 0.5 - cy)                                         <L 109>
        var_46 = wp::mul(var_32, var_45);
        var_47 = wp::sub(var_46, var_20);
        var_48 = wp::mul(var_35, var_47);
        // bottom = -inv_fy_znear * (sensor_h * 0.5 + cy)                                     <L 110>
        var_49 = wp::neg(var_35);
        var_51 = wp::mul(var_32, var_50);
        var_52 = wp::add(var_51, var_20);
        var_53 = wp::mul(var_49, var_52);
    }
    if (!var_12) {
        // fovy_rad = fovy * wp.static(wp.pi / 180.0)                                         <L 112>
        var_55 = wp::mul(var_fovy, var_54);
        // half_height = znear * wp.tan(0.5 * fovy_rad)                                       <L 113>
        var_57 = wp::mul(var_56, var_55);
        var_58 = wp::tan(var_57);
        var_59 = wp::mul(var_znear, var_58);
        // half_width = half_height * aspect                                                  <L 114>
        var_60 = wp::mul(var_59, var_8);
        // left = -half_width                                                                 <L 115>
        var_61 = wp::neg(var_60);
        // right = half_width                                                                 <L 116>
        var_62 = wp::copy(var_60);
        // top = half_height                                                                  <L 117>
        var_63 = wp::copy(var_59);
        // bottom = -half_height                                                              <L 118>
        var_64 = wp::neg(var_59);
    }
    var_65 = wp::where(var_12, var_32, var_10);
    var_66 = wp::where(var_12, var_40, var_61);
    var_67 = wp::where(var_12, var_44, var_62);
    var_68 = wp::where(var_12, var_48, var_63);
    var_69 = wp::where(var_12, var_53, var_64);
    // u = (float(px) + 0.5) / float(img_w)                                                   <L 120>
    var_70 = wp::float(var_px);
    var_72 = wp::add(var_70, var_71);
    var_73 = wp::float(var_img_w);
    var_74 = wp::div(var_72, var_73);
    // v = (float(py) + 0.5) / float(img_h)                                                   <L 121>
    var_75 = wp::float(var_py);
    var_77 = wp::add(var_75, var_76);
    var_78 = wp::float(var_img_h);
    var_79 = wp::div(var_77, var_78);
    // x = left + (right - left) * u                                                          <L 122>
    var_80 = wp::sub(var_67, var_66);
    var_81 = wp::mul(var_80, var_74);
    var_82 = wp::add(var_66, var_81);
    // y = top + (bottom - top) * v                                                           <L 123>
    var_83 = wp::sub(var_69, var_68);
    var_84 = wp::mul(var_83, var_79);
    var_85 = wp::add(var_68, var_84);
    // return wp.normalize(wp.vec3(x, y, -znear))                                             <L 125>
    var_86 = wp::neg(var_znear);
    var_87 = wp::vec_t<3, wp::float32>(var_82, var_85, var_86);
    var_88 = wp::normalize(var_87);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_88 += adj_ret;
    wp::adj_normalize(var_87, var_88, adj_87, adj_88);
    wp::adj_vec_t(var_82, var_85, var_86, adj_82, adj_85, adj_86, adj_87);
    wp::adj_neg(var_znear, adj_znear, adj_86);
    // adj: return wp.normalize(wp.vec3(x, y, -znear))                                        <L 125>
    wp::adj_add(var_68, var_84, adj_68, adj_84, adj_85);
    wp::adj_mul(var_83, var_79, adj_83, adj_79, adj_84);
    wp::adj_sub(var_69, var_68, adj_69, adj_68, adj_83);
    // adj: y = top + (bottom - top) * v                                                      <L 123>
    wp::adj_add(var_66, var_81, adj_66, adj_81, adj_82);
    wp::adj_mul(var_80, var_74, adj_80, adj_74, adj_81);
    wp::adj_sub(var_67, var_66, adj_67, adj_66, adj_80);
    // adj: x = left + (right - left) * u                                                     <L 122>
    wp::adj_div(var_77, var_78, var_79, adj_77, adj_78, adj_79);
    wp::adj_float(var_img_h, adj_img_h, adj_78);
    wp::adj_add(var_75, var_76, adj_75, adj_76, adj_77);
    wp::adj_float(var_py, adj_py, adj_75);
    // adj: v = (float(py) + 0.5) / float(img_h)                                              <L 121>
    wp::adj_div(var_72, var_73, var_74, adj_72, adj_73, adj_74);
    wp::adj_float(var_img_w, adj_img_w, adj_73);
    wp::adj_add(var_70, var_71, adj_70, adj_71, adj_72);
    wp::adj_float(var_px, adj_px, adj_70);
    // adj: u = (float(px) + 0.5) / float(img_w)                                              <L 120>
    wp::adj_where(var_12, var_53, var_64, adj_12, adj_53, adj_64, adj_69);
    wp::adj_where(var_12, var_48, var_63, adj_12, adj_48, adj_63, adj_68);
    wp::adj_where(var_12, var_44, var_62, adj_12, adj_44, adj_62, adj_67);
    wp::adj_where(var_12, var_40, var_61, adj_12, adj_40, adj_61, adj_66);
    wp::adj_where(var_12, var_32, var_10, adj_12, adj_32, adj_10, adj_65);
    if (!var_12) {
        wp::adj_neg(var_59, adj_59, adj_64);
        // adj: bottom = -half_height                                                         <L 118>
        wp::adj_copy(var_59, adj_59, adj_63);
        // adj: top = half_height                                                             <L 117>
        wp::adj_copy(var_60, adj_60, adj_62);
        // adj: right = half_width                                                            <L 116>
        wp::adj_neg(var_60, adj_60, adj_61);
        // adj: left = -half_width                                                            <L 115>
        wp::adj_mul(var_59, var_8, adj_59, adj_8, adj_60);
        // adj: half_width = half_height * aspect                                             <L 114>
        wp::adj_mul(var_znear, var_58, adj_znear, adj_58, adj_59);
        wp::adj_tan(var_57, adj_57, adj_58);
        wp::adj_mul(var_56, var_55, adj_56, adj_55, adj_57);
        // adj: half_height = znear * wp.tan(0.5 * fovy_rad)                                  <L 113>
        wp::adj_mul(var_fovy, var_54, adj_fovy, adj_54, adj_55);
        // adj: fovy_rad = fovy * wp.static(wp.pi / 180.0)                                    <L 112>
    }
    if (var_12) {
        wp::adj_mul(var_49, var_52, adj_49, adj_52, adj_53);
        wp::adj_add(var_51, var_20, adj_51, adj_20, adj_52);
        wp::adj_mul(var_32, var_50, adj_32, adj_50, adj_51);
        wp::adj_neg(var_35, adj_35, adj_49);
        // adj: bottom = -inv_fy_znear * (sensor_h * 0.5 + cy)                                <L 110>
        wp::adj_mul(var_35, var_47, adj_35, adj_47, adj_48);
        wp::adj_sub(var_46, var_20, adj_46, adj_20, adj_47);
        wp::adj_mul(var_32, var_45, adj_32, adj_45, adj_46);
        // adj: top = inv_fy_znear * (sensor_h * 0.5 - cy)                                    <L 109>
        wp::adj_mul(var_34, var_43, adj_34, adj_43, adj_44);
        wp::adj_add(var_42, var_18, adj_42, adj_18, adj_43);
        wp::adj_mul(var_33, var_41, adj_33, adj_41, adj_42);
        // adj: right = inv_fx_znear * (sensor_w * 0.5 + cx)                                  <L 108>
        wp::adj_mul(var_36, var_39, adj_36, adj_39, adj_40);
        wp::adj_sub(var_38, var_18, adj_38, adj_18, adj_39);
        wp::adj_mul(var_33, var_37, adj_33, adj_37, adj_38);
        wp::adj_neg(var_34, adj_34, adj_36);
        // adj: left = -inv_fx_znear * (sensor_w * 0.5 - cx)                                  <L 107>
        wp::adj_div(var_znear, var_16, var_35, adj_znear, adj_16, adj_35);
        // adj: inv_fy_znear = znear / fy                                                     <L 106>
        wp::adj_div(var_znear, var_14, var_34, adj_znear, adj_14, adj_34);
        // adj: inv_fx_znear = znear / fx                                                     <L 105>
        wp::adj_where(var_27, var_22, var_31, adj_27, adj_22, adj_31, adj_33);
        wp::adj_where(var_27, var_28, var_10, adj_27, adj_28, adj_10, adj_32);
        if (!var_27) {
            wp::adj_where(var_29, var_30, var_22, adj_29, adj_30, adj_22, adj_31);
            if (var_29) {
                wp::adj_mul(var_10, var_25, adj_10, adj_25, adj_30);
                // adj: sensor_w = sensor_h * target_aspect                                   <L 103>
            }
            // adj: elif target_aspect < sensor_aspect:                                       <L 102>
        }
        if (var_27) {
            wp::adj_div(var_22, var_25, var_28, adj_22, adj_25, adj_28);
            // adj: sensor_h = sensor_w / target_aspect                                       <L 101>
        }
        // adj: if target_aspect > sensor_aspect:                                             <L 100>
        wp::adj_div(var_22, var_10, var_26, adj_22, adj_10, adj_26);
        // adj: sensor_aspect = sensor_w / sensor_h                                           <L 99>
        wp::adj_div(var_23, var_24, var_25, adj_23, adj_24, adj_25);
        wp::adj_float(var_img_h, adj_img_h, adj_24);
        wp::adj_float(var_img_w, adj_img_w, adj_23);
        // adj: target_aspect = float(img_w) / float(img_h)                                   <L 98>
        wp::adj_extract(var_sensorsize, var_21, adj_sensorsize, adj_21, adj_22);
        // adj: sensor_w = sensorsize[0]                                                      <L 96>
        wp::adj_extract(var_intrinsic, var_19, adj_intrinsic, adj_19, adj_20);
        // adj: cy = intrinsic[3]                                                             <L 95>
        wp::adj_extract(var_intrinsic, var_17, adj_intrinsic, adj_17, adj_18);
        // adj: cx = intrinsic[2]                                                             <L 94>
        wp::adj_extract(var_intrinsic, var_15, adj_intrinsic, adj_15, adj_16);
        // adj: fy = intrinsic[1]                                                             <L 93>
        wp::adj_extract(var_intrinsic, var_13, adj_intrinsic, adj_13, adj_14);
        // adj: fx = intrinsic[0]                                                             <L 92>
    }
    // adj: if sensor_h != 0.0:                                                               <L 91>
    wp::adj_extract(var_sensorsize, var_9, adj_sensorsize, adj_9, adj_10);
    // adj: sensor_h = sensorsize[1]                                                          <L 88>
    wp::adj_div(var_6, var_7, var_8, adj_6, adj_7, adj_8);
    wp::adj_float(var_img_h, adj_img_h, adj_7);
    wp::adj_float(var_img_w, adj_img_w, adj_6);
    // adj: aspect = float(img_w) / float(img_h)                                              <L 87>
    if (var_1) {
        label0:;
        adj_5 += adj_ret;
        wp::adj_vec_t(var_2, var_3, var_4, adj_2, adj_3, adj_4, adj_5);
        // adj: return wp.vec3(0.0, 0.0, -1.0)                                                <L 85>
    }
    // adj: if projection == ProjectionType.ORTHOGRAPHIC:                                     <L 84>
    // adj: def compute_ray(                                                                  <L 67>
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::quat_t<wp::float32> & adj_ret)
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
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::quat_t<wp::float32> adj_11 = {};
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
    goto label0;
    //---------
    // reverse
    label0:;
    adj_11 += adj_ret;
    wp::adj_quat_t(var_1, var_4, var_7, var_10, adj_1, adj_4, adj_7, adj_10, adj_11);
    wp::adj_neg(var_9, adj_9, adj_10);
    wp::adj_extract(var_quat, var_8, adj_quat, adj_8, adj_9);
    wp::adj_neg(var_6, adj_6, adj_7);
    wp::adj_extract(var_quat, var_5, adj_quat, adj_5, adj_6);
    wp::adj_neg(var_3, adj_3, adj_4);
    wp::adj_extract(var_quat, var_2, adj_quat, adj_2, adj_3);
    wp::adj_extract(var_quat, var_0, adj_quat, adj_0, adj_1);
    // adj: return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])                             <L 117>
    // adj: def quat_inv(quat: wp.quat) -> wp.quat:                                           <L 116>
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE void adj_mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v,
    wp::quat_t<wp::float32> & adj_u,
    wp::quat_t<wp::float32> & adj_v,
    wp::quat_t<wp::float32> & adj_ret)
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
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::int32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::int32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::int32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::int32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::int32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::int32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::int32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::int32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::int32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::int32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::int32 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::int32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::float32 adj_70 = {};
    wp::int32 adj_71 = {};
    wp::float32 adj_72 = {};
    wp::float32 adj_73 = {};
    wp::int32 adj_74 = {};
    wp::float32 adj_75 = {};
    wp::int32 adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::int32 adj_80 = {};
    wp::float32 adj_81 = {};
    wp::int32 adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::int32 adj_86 = {};
    wp::float32 adj_87 = {};
    wp::int32 adj_88 = {};
    wp::float32 adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float32 adj_91 = {};
    wp::quat_t<wp::float32> adj_92 = {};
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
    goto label0;
    //---------
    // reverse
    label0:;
    adj_92 += adj_ret;
    wp::adj_quat_t(var_22, var_45, var_68, var_91, adj_22, adj_45, adj_68, adj_91, adj_92);
    wp::adj_add(var_85, var_90, adj_85, adj_90, adj_91);
    wp::adj_mul(var_87, var_89, adj_87, adj_89, adj_90);
    wp::adj_extract(var_v, var_88, adj_v, adj_88, adj_89);
    wp::adj_extract(var_u, var_86, adj_u, adj_86, adj_87);
    wp::adj_sub(var_79, var_84, adj_79, adj_84, adj_85);
    wp::adj_mul(var_81, var_83, adj_81, adj_83, adj_84);
    wp::adj_extract(var_v, var_82, adj_v, adj_82, adj_83);
    wp::adj_extract(var_u, var_80, adj_u, adj_80, adj_81);
    wp::adj_add(var_73, var_78, adj_73, adj_78, adj_79);
    wp::adj_mul(var_75, var_77, adj_75, adj_77, adj_78);
    wp::adj_extract(var_v, var_76, adj_v, adj_76, adj_77);
    wp::adj_extract(var_u, var_74, adj_u, adj_74, adj_75);
    wp::adj_mul(var_70, var_72, adj_70, adj_72, adj_73);
    wp::adj_extract(var_v, var_71, adj_v, adj_71, adj_72);
    wp::adj_extract(var_u, var_69, adj_u, adj_69, adj_70);
    // adj: u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],                            <L 29>
    wp::adj_add(var_62, var_67, adj_62, adj_67, adj_68);
    wp::adj_mul(var_64, var_66, adj_64, adj_66, adj_67);
    wp::adj_extract(var_v, var_65, adj_v, adj_65, adj_66);
    wp::adj_extract(var_u, var_63, adj_u, adj_63, adj_64);
    wp::adj_add(var_56, var_61, adj_56, adj_61, adj_62);
    wp::adj_mul(var_58, var_60, adj_58, adj_60, adj_61);
    wp::adj_extract(var_v, var_59, adj_v, adj_59, adj_60);
    wp::adj_extract(var_u, var_57, adj_u, adj_57, adj_58);
    wp::adj_sub(var_50, var_55, adj_50, adj_55, adj_56);
    wp::adj_mul(var_52, var_54, adj_52, adj_54, adj_55);
    wp::adj_extract(var_v, var_53, adj_v, adj_53, adj_54);
    wp::adj_extract(var_u, var_51, adj_u, adj_51, adj_52);
    wp::adj_mul(var_47, var_49, adj_47, adj_49, adj_50);
    wp::adj_extract(var_v, var_48, adj_v, adj_48, adj_49);
    wp::adj_extract(var_u, var_46, adj_u, adj_46, adj_47);
    // adj: u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],                            <L 28>
    wp::adj_sub(var_39, var_44, adj_39, adj_44, adj_45);
    wp::adj_mul(var_41, var_43, adj_41, adj_43, adj_44);
    wp::adj_extract(var_v, var_42, adj_v, adj_42, adj_43);
    wp::adj_extract(var_u, var_40, adj_u, adj_40, adj_41);
    wp::adj_add(var_33, var_38, adj_33, adj_38, adj_39);
    wp::adj_mul(var_35, var_37, adj_35, adj_37, adj_38);
    wp::adj_extract(var_v, var_36, adj_v, adj_36, adj_37);
    wp::adj_extract(var_u, var_34, adj_u, adj_34, adj_35);
    wp::adj_add(var_27, var_32, adj_27, adj_32, adj_33);
    wp::adj_mul(var_29, var_31, adj_29, adj_31, adj_32);
    wp::adj_extract(var_v, var_30, adj_v, adj_30, adj_31);
    wp::adj_extract(var_u, var_28, adj_u, adj_28, adj_29);
    wp::adj_mul(var_24, var_26, adj_24, adj_26, adj_27);
    wp::adj_extract(var_v, var_25, adj_v, adj_25, adj_26);
    wp::adj_extract(var_u, var_23, adj_u, adj_23, adj_24);
    // adj: u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],                            <L 27>
    wp::adj_sub(var_16, var_21, adj_16, adj_21, adj_22);
    wp::adj_mul(var_18, var_20, adj_18, adj_20, adj_21);
    wp::adj_extract(var_v, var_19, adj_v, adj_19, adj_20);
    wp::adj_extract(var_u, var_17, adj_u, adj_17, adj_18);
    wp::adj_sub(var_10, var_15, adj_10, adj_15, adj_16);
    wp::adj_mul(var_12, var_14, adj_12, adj_14, adj_15);
    wp::adj_extract(var_v, var_13, adj_v, adj_13, adj_14);
    wp::adj_extract(var_u, var_11, adj_u, adj_11, adj_12);
    wp::adj_sub(var_4, var_9, adj_4, adj_9, adj_10);
    wp::adj_mul(var_6, var_8, adj_6, adj_8, adj_9);
    wp::adj_extract(var_v, var_7, adj_v, adj_7, adj_8);
    wp::adj_extract(var_u, var_5, adj_u, adj_5, adj_6);
    wp::adj_mul(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_extract(var_v, var_2, adj_v, adj_2, adj_3);
    wp::adj_extract(var_u, var_0, adj_u, adj_0, adj_1);
    // adj: u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],                            <L 26>
    // adj: return wp.quat(                                                                   <L 25>
    // adj: def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:                                  <L 24>
    return;
}


// /oscar/scratch/sboughan/venvs/se7-ogb/lib/python3.12/site-packages/mujoco_warp/_src/math.py:120
static CUDA_CALLABLE void adj_inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<10, wp::float32> & adj_i,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
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
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::float32 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::int32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::int32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::int32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::int32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::int32 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::int32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::int32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::int32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::int32 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::int32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::float32 adj_70 = {};
    wp::int32 adj_71 = {};
    wp::float32 adj_72 = {};
    wp::float32 adj_73 = {};
    wp::float32 adj_74 = {};
    wp::int32 adj_75 = {};
    wp::float32 adj_76 = {};
    wp::int32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::float32 adj_80 = {};
    wp::int32 adj_81 = {};
    wp::float32 adj_82 = {};
    wp::int32 adj_83 = {};
    wp::float32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::float32 adj_86 = {};
    wp::int32 adj_87 = {};
    wp::float32 adj_88 = {};
    wp::int32 adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float32 adj_91 = {};
    wp::int32 adj_92 = {};
    wp::float32 adj_93 = {};
    wp::int32 adj_94 = {};
    wp::float32 adj_95 = {};
    wp::float32 adj_96 = {};
    wp::float32 adj_97 = {};
    wp::int32 adj_98 = {};
    wp::float32 adj_99 = {};
    wp::int32 adj_100 = {};
    wp::float32 adj_101 = {};
    wp::float32 adj_102 = {};
    wp::float32 adj_103 = {};
    wp::int32 adj_104 = {};
    wp::float32 adj_105 = {};
    wp::int32 adj_106 = {};
    wp::float32 adj_107 = {};
    wp::float32 adj_108 = {};
    wp::int32 adj_109 = {};
    wp::float32 adj_110 = {};
    wp::int32 adj_111 = {};
    wp::float32 adj_112 = {};
    wp::float32 adj_113 = {};
    wp::float32 adj_114 = {};
    wp::int32 adj_115 = {};
    wp::float32 adj_116 = {};
    wp::int32 adj_117 = {};
    wp::float32 adj_118 = {};
    wp::float32 adj_119 = {};
    wp::float32 adj_120 = {};
    wp::int32 adj_121 = {};
    wp::float32 adj_122 = {};
    wp::int32 adj_123 = {};
    wp::float32 adj_124 = {};
    wp::float32 adj_125 = {};
    wp::int32 adj_126 = {};
    wp::float32 adj_127 = {};
    wp::int32 adj_128 = {};
    wp::float32 adj_129 = {};
    wp::float32 adj_130 = {};
    wp::float32 adj_131 = {};
    wp::int32 adj_132 = {};
    wp::float32 adj_133 = {};
    wp::int32 adj_134 = {};
    wp::float32 adj_135 = {};
    wp::float32 adj_136 = {};
    wp::float32 adj_137 = {};
    wp::vec_t<6, wp::float32> adj_138 = {};
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
    goto label0;
    //---------
    // reverse
    label0:;
    adj_138 += adj_ret;
    wp::adj_vec_t({var_28, var_57, var_86, var_103, var_120, var_137}, {&adj_28, &adj_57, &adj_86, &adj_103, &adj_120, &adj_137}, adj_138);
    wp::adj_add(var_131, var_136, adj_131, adj_136, adj_137);
    wp::adj_mul(var_133, var_135, adj_133, adj_135, adj_136);
    wp::adj_extract(var_v, var_134, adj_v, adj_134, adj_135);
    wp::adj_extract(var_i, var_132, adj_i, adj_132, adj_133);
    wp::adj_sub(var_125, var_130, adj_125, adj_130, adj_131);
    wp::adj_mul(var_127, var_129, adj_127, adj_129, adj_130);
    wp::adj_extract(var_v, var_128, adj_v, adj_128, adj_129);
    wp::adj_extract(var_i, var_126, adj_i, adj_126, adj_127);
    wp::adj_mul(var_122, var_124, adj_122, adj_124, adj_125);
    wp::adj_extract(var_v, var_123, adj_v, adj_123, adj_124);
    wp::adj_extract(var_i, var_121, adj_i, adj_121, adj_122);
    // adj: i[7] * v[0] - i[6] * v[1] + i[9] * v[5],                                          <L 129>
    wp::adj_add(var_114, var_119, adj_114, adj_119, adj_120);
    wp::adj_mul(var_116, var_118, adj_116, adj_118, adj_119);
    wp::adj_extract(var_v, var_117, adj_v, adj_117, adj_118);
    wp::adj_extract(var_i, var_115, adj_i, adj_115, adj_116);
    wp::adj_sub(var_108, var_113, adj_108, adj_113, adj_114);
    wp::adj_mul(var_110, var_112, adj_110, adj_112, adj_113);
    wp::adj_extract(var_v, var_111, adj_v, adj_111, adj_112);
    wp::adj_extract(var_i, var_109, adj_i, adj_109, adj_110);
    wp::adj_mul(var_105, var_107, adj_105, adj_107, adj_108);
    wp::adj_extract(var_v, var_106, adj_v, adj_106, adj_107);
    wp::adj_extract(var_i, var_104, adj_i, adj_104, adj_105);
    // adj: i[6] * v[2] - i[8] * v[0] + i[9] * v[4],                                          <L 128>
    wp::adj_add(var_97, var_102, adj_97, adj_102, adj_103);
    wp::adj_mul(var_99, var_101, adj_99, adj_101, adj_102);
    wp::adj_extract(var_v, var_100, adj_v, adj_100, adj_101);
    wp::adj_extract(var_i, var_98, adj_i, adj_98, adj_99);
    wp::adj_sub(var_91, var_96, adj_91, adj_96, adj_97);
    wp::adj_mul(var_93, var_95, adj_93, adj_95, adj_96);
    wp::adj_extract(var_v, var_94, adj_v, adj_94, adj_95);
    wp::adj_extract(var_i, var_92, adj_i, adj_92, adj_93);
    wp::adj_mul(var_88, var_90, adj_88, adj_90, adj_91);
    wp::adj_extract(var_v, var_89, adj_v, adj_89, adj_90);
    wp::adj_extract(var_i, var_87, adj_i, adj_87, adj_88);
    // adj: i[8] * v[1] - i[7] * v[2] + i[9] * v[3],                                          <L 127>
    wp::adj_add(var_80, var_85, adj_80, adj_85, adj_86);
    wp::adj_mul(var_82, var_84, adj_82, adj_84, adj_85);
    wp::adj_extract(var_v, var_83, adj_v, adj_83, adj_84);
    wp::adj_extract(var_i, var_81, adj_i, adj_81, adj_82);
    wp::adj_sub(var_74, var_79, adj_74, adj_79, adj_80);
    wp::adj_mul(var_76, var_78, adj_76, adj_78, adj_79);
    wp::adj_extract(var_v, var_77, adj_v, adj_77, adj_78);
    wp::adj_extract(var_i, var_75, adj_i, adj_75, adj_76);
    wp::adj_add(var_68, var_73, adj_68, adj_73, adj_74);
    wp::adj_mul(var_70, var_72, adj_70, adj_72, adj_73);
    wp::adj_extract(var_v, var_71, adj_v, adj_71, adj_72);
    wp::adj_extract(var_i, var_69, adj_i, adj_69, adj_70);
    wp::adj_add(var_62, var_67, adj_62, adj_67, adj_68);
    wp::adj_mul(var_64, var_66, adj_64, adj_66, adj_67);
    wp::adj_extract(var_v, var_65, adj_v, adj_65, adj_66);
    wp::adj_extract(var_i, var_63, adj_i, adj_63, adj_64);
    wp::adj_mul(var_59, var_61, adj_59, adj_61, adj_62);
    wp::adj_extract(var_v, var_60, adj_v, adj_60, adj_61);
    wp::adj_extract(var_i, var_58, adj_i, adj_58, adj_59);
    // adj: i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],              <L 126>
    wp::adj_sub(var_51, var_56, adj_51, adj_56, adj_57);
    wp::adj_mul(var_53, var_55, adj_53, adj_55, adj_56);
    wp::adj_extract(var_v, var_54, adj_v, adj_54, adj_55);
    wp::adj_extract(var_i, var_52, adj_i, adj_52, adj_53);
    wp::adj_add(var_45, var_50, adj_45, adj_50, adj_51);
    wp::adj_mul(var_47, var_49, adj_47, adj_49, adj_50);
    wp::adj_extract(var_v, var_48, adj_v, adj_48, adj_49);
    wp::adj_extract(var_i, var_46, adj_i, adj_46, adj_47);
    wp::adj_add(var_39, var_44, adj_39, adj_44, adj_45);
    wp::adj_mul(var_41, var_43, adj_41, adj_43, adj_44);
    wp::adj_extract(var_v, var_42, adj_v, adj_42, adj_43);
    wp::adj_extract(var_i, var_40, adj_i, adj_40, adj_41);
    wp::adj_add(var_33, var_38, adj_33, adj_38, adj_39);
    wp::adj_mul(var_35, var_37, adj_35, adj_37, adj_38);
    wp::adj_extract(var_v, var_36, adj_v, adj_36, adj_37);
    wp::adj_extract(var_i, var_34, adj_i, adj_34, adj_35);
    wp::adj_mul(var_30, var_32, adj_30, adj_32, adj_33);
    wp::adj_extract(var_v, var_31, adj_v, adj_31, adj_32);
    wp::adj_extract(var_i, var_29, adj_i, adj_29, adj_30);
    // adj: i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],              <L 125>
    wp::adj_add(var_22, var_27, adj_22, adj_27, adj_28);
    wp::adj_mul(var_24, var_26, adj_24, adj_26, adj_27);
    wp::adj_extract(var_v, var_25, adj_v, adj_25, adj_26);
    wp::adj_extract(var_i, var_23, adj_i, adj_23, adj_24);
    wp::adj_sub(var_16, var_21, adj_16, adj_21, adj_22);
    wp::adj_mul(var_18, var_20, adj_18, adj_20, adj_21);
    wp::adj_extract(var_v, var_19, adj_v, adj_19, adj_20);
    wp::adj_extract(var_i, var_17, adj_i, adj_17, adj_18);
    wp::adj_add(var_10, var_15, adj_10, adj_15, adj_16);
    wp::adj_mul(var_12, var_14, adj_12, adj_14, adj_15);
    wp::adj_extract(var_v, var_13, adj_v, adj_13, adj_14);
    wp::adj_extract(var_i, var_11, adj_i, adj_11, adj_12);
    wp::adj_add(var_4, var_9, adj_4, adj_9, adj_10);
    wp::adj_mul(var_6, var_8, adj_6, adj_8, adj_9);
    wp::adj_extract(var_v, var_7, adj_v, adj_7, adj_8);
    wp::adj_extract(var_i, var_5, adj_i, adj_5, adj_6);
    wp::adj_mul(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_extract(var_v, var_2, adj_v, adj_2, adj_3);
    wp::adj_extract(var_i, var_0, adj_i, adj_0, adj_1);
    // adj: i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],              <L 124>
    // adj: return wp.spatial_vector(                                                         <L 123>
    // adj: def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:         <L 121>
    return;
}



extern "C" __global__ void _copy_qpos0_to_qpos_32739b15_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<wp::float32> var_qpos_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def _copy_qpos0_to_qpos(                                                               <L 2974>
        // worldid, i = wp.tid()                                                                  <L 2978>
        builtin_tid2d(var_0, var_1);
        // qpos0_id = worldid % qpos0.shape[0]                                                    <L 2979>
        var_2 = &(var_qpos0.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // qpos_out[worldid, i] = qpos0[qpos0_id, i]                                              <L 2980>
        var_7 = wp::address(var_qpos0, var_6, var_1);
        var_8 = wp::load(var_7);
        wp::array_store(var_qpos_out, var_0, var_1, var_8);
}



extern "C" __global__ void _copy_qpos0_to_qpos_32739b15_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<wp::float32> var_qpos_out,
    wp::array_t<wp::float32> adj_qpos0,
    wp::array_t<wp::float32> adj_qpos_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        //---------
        // forward
        // def _copy_qpos0_to_qpos(                                                               <L 2974>
        // worldid, i = wp.tid()                                                                  <L 2978>
        builtin_tid2d(var_0, var_1);
        // qpos0_id = worldid % qpos0.shape[0]                                                    <L 2979>
        var_2 = &(var_qpos0.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // qpos_out[worldid, i] = qpos0[qpos0_id, i]                                              <L 2980>
        var_7 = wp::address(var_qpos0, var_6, var_1);
        var_8 = wp::load(var_7);
        // wp::array_store(var_qpos_out, var_0, var_1, var_8);
        //---------
        // reverse
        wp::adj_array_store(var_qpos_out, var_0, var_1, var_8, adj_qpos_out, adj_0, adj_1, adj_7);
        wp::adj_address(var_qpos0, var_6, var_1, adj_qpos0, adj_6, adj_1, adj_7);
        // adj: qpos_out[worldid, i] = qpos0[qpos0_id, i]                                         <L 2980>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_qpos0.shape = adj_2;
        // adj: qpos0_id = worldid % qpos0.shape[0]                                               <L 2979>
        // adj: worldid, i = wp.tid()                                                             <L 2978>
        // adj: def _copy_qpos0_to_qpos(                                                          <L 2974>
        return;
}



extern "C" __global__ void _copy_actuator_moment_4af7079d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_act_moment_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _copy_actuator_moment(                                                             <L 3387>
        // worldid = wp.tid()                                                                     <L 3395>
        var_0 = builtin_tid1d();
        // nv = act_moment_vec_out.shape[1]                                                       <L 3396>
        var_1 = &(var_act_moment_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 3397>
        var_5 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // act_moment_vec_out[worldid, i] = 0.0                                               <L 3398>
            wp::array_store(var_act_moment_vec_out, var_0, var_6, var_7);
            goto start_for_0;
        end_for_0:;
        // rownnz = moment_rownnz_in[worldid, actid_target]                                       <L 3399>
        var_8 = wp::address(var_moment_rownnz_in, var_0, var_actid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = moment_rowadr_in[worldid, actid_target]                                       <L 3400>
        var_11 = wp::address(var_moment_rowadr_in, var_0, var_actid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 3401>
        var_14 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_14) == 0) goto end_for_2;
            var_15 = wp::iter_next(var_14);
            // sparseid = rowadr + i                                                              <L 3402>
            var_16 = wp::add(var_12, var_15);
            // col = moment_colind_in[worldid, sparseid]                                          <L 3403>
            var_17 = wp::address(var_moment_colind_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // act_moment_vec_out[worldid, col] = actuator_moment_in[worldid, sparseid]           <L 3404>
            var_20 = wp::address(var_actuator_moment_in, var_0, var_16);
            var_21 = wp::load(var_20);
            wp::array_store(var_act_moment_vec_out, var_0, var_18, var_21);
            goto start_for_2;
        end_for_2:;
}



extern "C" __global__ void _copy_actuator_moment_4af7079d_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_act_moment_vec_out,
    wp::int32 adj_actid_target,
    wp::array_t<wp::int32> adj_moment_rownnz_in,
    wp::array_t<wp::int32> adj_moment_rowadr_in,
    wp::array_t<wp::int32> adj_moment_colind_in,
    wp::array_t<wp::float32> adj_actuator_moment_in,
    wp::array_t<wp::float32> adj_act_moment_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::shape_t adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::range_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::range_t adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        //---------
        // forward
        // def _copy_actuator_moment(                                                             <L 3387>
        // worldid = wp.tid()                                                                     <L 3395>
        var_0 = builtin_tid1d();
        // nv = act_moment_vec_out.shape[1]                                                       <L 3396>
        var_1 = &(var_act_moment_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 3397>
        var_5 = wp::range(var_3);
        // rownnz = moment_rownnz_in[worldid, actid_target]                                       <L 3399>
        var_8 = wp::address(var_moment_rownnz_in, var_0, var_actid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = moment_rowadr_in[worldid, actid_target]                                       <L 3400>
        var_11 = wp::address(var_moment_rowadr_in, var_0, var_actid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 3401>
        var_14 = wp::range(var_9);
        //---------
        // reverse
        var_14 = wp::iter_reverse(var_14);
        start_for_2:;
            if (iter_cmp(var_14) == 0) goto end_for_2;
            var_15 = wp::iter_next(var_14);
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
            // sparseid = rowadr + i                                                              <L 3402>
            var_16 = wp::add(var_12, var_15);
            // col = moment_colind_in[worldid, sparseid]                                          <L 3403>
            var_17 = wp::address(var_moment_colind_in, var_0, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // act_moment_vec_out[worldid, col] = actuator_moment_in[worldid, sparseid]           <L 3404>
            var_20 = wp::address(var_actuator_moment_in, var_0, var_16);
            var_21 = wp::load(var_20);
            // wp::array_store(var_act_moment_vec_out, var_0, var_18, var_21);
            wp::adj_array_store(var_act_moment_vec_out, var_0, var_18, var_21, adj_act_moment_vec_out, adj_0, adj_18, adj_20);
            wp::adj_address(var_actuator_moment_in, var_0, var_16, adj_actuator_moment_in, adj_0, adj_16, adj_20);
            // adj: act_moment_vec_out[worldid, col] = actuator_moment_in[worldid, sparseid]      <L 3404>
            wp::adj_copy(var_19, adj_17, adj_18);
            wp::adj_address(var_moment_colind_in, var_0, var_16, adj_moment_colind_in, adj_0, adj_16, adj_17);
            // adj: col = moment_colind_in[worldid, sparseid]                                     <L 3403>
            wp::adj_add(var_12, var_15, adj_12, adj_15, adj_16);
            // adj: sparseid = rowadr + i                                                         <L 3402>
        	goto start_for_2;
        end_for_2:;
        // adj: for i in range(rownnz):                                                           <L 3401>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_moment_rowadr_in, var_0, var_actid_target, adj_moment_rowadr_in, adj_0, adj_actid_target, adj_11);
        // adj: rowadr = moment_rowadr_in[worldid, actid_target]                                  <L 3400>
        wp::adj_copy(var_10, adj_8, adj_9);
        wp::adj_address(var_moment_rownnz_in, var_0, var_actid_target, adj_moment_rownnz_in, adj_0, adj_actid_target, adj_8);
        // adj: rownnz = moment_rownnz_in[worldid, actid_target]                                  <L 3399>
        var_5 = wp::iter_reverse(var_5);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
        	adj_7 = {};
            // act_moment_vec_out[worldid, i] = 0.0                                               <L 3398>
            // wp::array_store(var_act_moment_vec_out, var_0, var_6, var_7);
            wp::adj_array_store(var_act_moment_vec_out, var_0, var_6, var_7, adj_act_moment_vec_out, adj_0, adj_6, adj_7);
            // adj: act_moment_vec_out[worldid, i] = 0.0                                          <L 3398>
        	goto start_for_0;
        end_for_0:;
        // adj: for i in range(nv):                                                               <L 3397>
        adj_act_moment_vec_out.shape = adj_1;
        // adj: nv = act_moment_vec_out.shape[1]                                                  <L 3396>
        // adj: worldid = wp.tid()                                                                <L 3395>
        // adj: def _copy_actuator_moment(                                                        <L 3387>
        return;
}



extern "C" __global__ void _copy_tendon_length0_5230a976_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_tendon_length0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def _copy_tendon_length0(                                                              <L 2984>
        // worldid, tenid = wp.tid()                                                              <L 2988>
        builtin_tid2d(var_0, var_1);
        // tendon_length0_id = worldid % tendon_length0_out.shape[0]                              <L 2989>
        var_2 = &(var_tendon_length0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // tendon_length0_out[tendon_length0_id, tenid] = ten_length_in[worldid, tenid]           <L 2990>
        var_7 = wp::address(var_ten_length_in, var_0, var_1);
        var_8 = wp::load(var_7);
        wp::array_store(var_tendon_length0_out, var_6, var_1, var_8);
}



extern "C" __global__ void _copy_tendon_length0_5230a976_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_tendon_length0_out,
    wp::array_t<wp::float32> adj_ten_length_in,
    wp::array_t<wp::float32> adj_tendon_length0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        //---------
        // forward
        // def _copy_tendon_length0(                                                              <L 2984>
        // worldid, tenid = wp.tid()                                                              <L 2988>
        builtin_tid2d(var_0, var_1);
        // tendon_length0_id = worldid % tendon_length0_out.shape[0]                              <L 2989>
        var_2 = &(var_tendon_length0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // tendon_length0_out[tendon_length0_id, tenid] = ten_length_in[worldid, tenid]           <L 2990>
        var_7 = wp::address(var_ten_length_in, var_0, var_1);
        var_8 = wp::load(var_7);
        // wp::array_store(var_tendon_length0_out, var_6, var_1, var_8);
        //---------
        // reverse
        wp::adj_array_store(var_tendon_length0_out, var_6, var_1, var_8, adj_tendon_length0_out, adj_6, adj_1, adj_7);
        wp::adj_address(var_ten_length_in, var_0, var_1, adj_ten_length_in, adj_0, adj_1, adj_7);
        // adj: tendon_length0_out[tendon_length0_id, tenid] = ten_length_in[worldid, tenid]      <L 2990>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_tendon_length0_out.shape = adj_2;
        // adj: tendon_length0_id = worldid % tendon_length0_out.shape[0]                         <L 2989>
        // adj: worldid, tenid = wp.tid()                                                         <L 2988>
        // adj: def _copy_tendon_length0(                                                         <L 2984>
        return;
}



extern "C" __global__ void _compute_cam_pos0_5993818e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_poscom0_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_mat0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::mat_t<3, 3, wp::float32>* var_27;
        wp::mat_t<3, 3, wp::float32> var_28;
        //---------
        // forward
        // def _compute_cam_pos0(                                                                 <L 3335>
        // worldid, camid = wp.tid()                                                              <L 3346>
        builtin_tid2d(var_0, var_1);
        // cam_pos0_id = worldid % cam_pos0_out.shape[0]                                          <L 3347>
        var_2 = &(var_cam_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = cam_bodyid[camid]                                                             <L 3348>
        var_7 = wp::address(var_cam_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = cam_targetbodyid[camid]                                                     <L 3349>
        var_10 = wp::address(var_cam_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // cam_xpos = cam_xpos_in[worldid, camid]                                                 <L 3350>
        var_13 = wp::address(var_cam_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // cam_pos0_out[cam_pos0_id, camid] = cam_xpos - xpos_in[worldid, bodyid]                 <L 3352>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        wp::array_store(var_cam_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 3353>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, targetid]       <L 3354>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            wp::array_store(var_cam_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, bodyid]       <L 3356>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            wp::array_store(var_cam_poscom0_out, var_6, var_1, var_25);
        }
        // cam_mat0_out[cam_pos0_id, camid] = cam_xmat_in[worldid, camid]                         <L 3357>
        var_27 = wp::address(var_cam_xmat_in, var_0, var_1);
        var_28 = wp::load(var_27);
        wp::array_store(var_cam_mat0_out, var_6, var_1, var_28);
}



extern "C" __global__ void _compute_cam_pos0_5993818e_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_poscom0_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_mat0_out,
    wp::array_t<wp::int32> adj_cam_bodyid,
    wp::array_t<wp::int32> adj_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_cam_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_cam_poscom0_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_cam_mat0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::mat_t<3, 3, wp::float32>* var_27;
        wp::mat_t<3, 3, wp::float32> var_28;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::int32 adj_19 = {};
        bool adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::mat_t<3, 3, wp::float32> adj_27 = {};
        wp::mat_t<3, 3, wp::float32> adj_28 = {};
        //---------
        // forward
        // def _compute_cam_pos0(                                                                 <L 3335>
        // worldid, camid = wp.tid()                                                              <L 3346>
        builtin_tid2d(var_0, var_1);
        // cam_pos0_id = worldid % cam_pos0_out.shape[0]                                          <L 3347>
        var_2 = &(var_cam_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = cam_bodyid[camid]                                                             <L 3348>
        var_7 = wp::address(var_cam_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = cam_targetbodyid[camid]                                                     <L 3349>
        var_10 = wp::address(var_cam_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // cam_xpos = cam_xpos_in[worldid, camid]                                                 <L 3350>
        var_13 = wp::address(var_cam_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // cam_pos0_out[cam_pos0_id, camid] = cam_xpos - xpos_in[worldid, bodyid]                 <L 3352>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        // wp::array_store(var_cam_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 3353>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, targetid]       <L 3354>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            // wp::array_store(var_cam_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, bodyid]       <L 3356>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            // wp::array_store(var_cam_poscom0_out, var_6, var_1, var_25);
        }
        // cam_mat0_out[cam_pos0_id, camid] = cam_xmat_in[worldid, camid]                         <L 3357>
        var_27 = wp::address(var_cam_xmat_in, var_0, var_1);
        var_28 = wp::load(var_27);
        // wp::array_store(var_cam_mat0_out, var_6, var_1, var_28);
        //---------
        // reverse
        wp::adj_array_store(var_cam_mat0_out, var_6, var_1, var_28, adj_cam_mat0_out, adj_6, adj_1, adj_27);
        wp::adj_address(var_cam_xmat_in, var_0, var_1, adj_cam_xmat_in, adj_0, adj_1, adj_27);
        // adj: cam_mat0_out[cam_pos0_id, camid] = cam_xmat_in[worldid, camid]                    <L 3357>
        if (!var_20) {
            wp::adj_array_store(var_cam_poscom0_out, var_6, var_1, var_25, adj_cam_poscom0_out, adj_6, adj_1, adj_25);
            wp::adj_sub(var_14, var_26, adj_14, adj_24, adj_25);
            wp::adj_address(var_subtree_com_in, var_0, var_8, adj_subtree_com_in, adj_0, adj_8, adj_24);
            // adj: cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, bodyid]  <L 3356>
        }
        if (var_20) {
            wp::adj_array_store(var_cam_poscom0_out, var_6, var_1, var_22, adj_cam_poscom0_out, adj_6, adj_1, adj_22);
            wp::adj_sub(var_14, var_23, adj_14, adj_21, adj_22);
            wp::adj_address(var_subtree_com_in, var_0, var_11, adj_subtree_com_in, adj_0, adj_11, adj_21);
            // adj: cam_poscom0_out[cam_pos0_id, camid] = cam_xpos - subtree_com_in[worldid, targetid]  <L 3354>
        }
        // adj: if targetid >= 0:                                                                 <L 3353>
        wp::adj_array_store(var_cam_pos0_out, var_6, var_1, var_17, adj_cam_pos0_out, adj_6, adj_1, adj_17);
        wp::adj_sub(var_14, var_18, adj_14, adj_16, adj_17);
        wp::adj_address(var_xpos_in, var_0, var_8, adj_xpos_in, adj_0, adj_8, adj_16);
        // adj: cam_pos0_out[cam_pos0_id, camid] = cam_xpos - xpos_in[worldid, bodyid]            <L 3352>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_cam_xpos_in, var_0, var_1, adj_cam_xpos_in, adj_0, adj_1, adj_13);
        // adj: cam_xpos = cam_xpos_in[worldid, camid]                                            <L 3350>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_cam_targetbodyid, var_1, adj_cam_targetbodyid, adj_1, adj_10);
        // adj: targetid = cam_targetbodyid[camid]                                                <L 3349>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_cam_bodyid, var_1, adj_cam_bodyid, adj_1, adj_7);
        // adj: bodyid = cam_bodyid[camid]                                                        <L 3348>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_cam_pos0_out.shape = adj_2;
        // adj: cam_pos0_id = worldid % cam_pos0_out.shape[0]                                     <L 3347>
        // adj: worldid, camid = wp.tid()                                                         <L 3346>
        // adj: def _compute_cam_pos0(                                                            <L 3335>
        return;
}



extern "C" __global__ void _build_rays_5e1b397e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_offset,
    wp::int32 var_img_w,
    wp::int32 var_img_h,
    wp::int32 var_projection,
    wp::float32 var_fovy,
    wp::vec_t<2, wp::float32> var_sensorsize,
    wp::vec_t<4, wp::float32> var_intrinsic,
    wp::float32 var_znear,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_out)
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
        wp::vec_t<3, wp::float32> var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        //---------
        // forward
        // def _build_rays(                                                                       <L 4067>
        // xid, yid = wp.tid()                                                                    <L 4080>
        builtin_tid2d(var_0, var_1);
        // ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                         <L 4081>
        // projection, fovy, sensorsize, intrinsic, img_w, img_h, xid, yid, znear                 <L 4082>
        var_2 = compute_ray_0(var_projection, var_fovy, var_sensorsize, var_intrinsic, var_img_w, var_img_h, var_0, var_1, var_znear);
        // ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                         <L 4081>
        var_3 = wp::add(var_offset, var_0);
        var_4 = wp::mul(var_1, var_img_w);
        var_5 = wp::add(var_3, var_4);
        wp::array_store(var_ray_out, var_5, var_2);
}



extern "C" __global__ void _build_rays_5e1b397e_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_offset,
    wp::int32 var_img_w,
    wp::int32 var_img_h,
    wp::int32 var_projection,
    wp::float32 var_fovy,
    wp::vec_t<2, wp::float32> var_sensorsize,
    wp::vec_t<4, wp::float32> var_intrinsic,
    wp::float32 var_znear,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_out,
    wp::int32 adj_offset,
    wp::int32 adj_img_w,
    wp::int32 adj_img_h,
    wp::int32 adj_projection,
    wp::float32 adj_fovy,
    wp::vec_t<2, wp::float32> adj_sensorsize,
    wp::vec_t<4, wp::float32> adj_intrinsic,
    wp::float32 adj_znear,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_ray_out)
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
        wp::vec_t<3, wp::float32> var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        //---------
        // forward
        // def _build_rays(                                                                       <L 4067>
        // xid, yid = wp.tid()                                                                    <L 4080>
        builtin_tid2d(var_0, var_1);
        // ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                         <L 4081>
        // projection, fovy, sensorsize, intrinsic, img_w, img_h, xid, yid, znear                 <L 4082>
        var_2 = compute_ray_0(var_projection, var_fovy, var_sensorsize, var_intrinsic, var_img_w, var_img_h, var_0, var_1, var_znear);
        // ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                         <L 4081>
        var_3 = wp::add(var_offset, var_0);
        var_4 = wp::mul(var_1, var_img_w);
        var_5 = wp::add(var_3, var_4);
        // wp::array_store(var_ray_out, var_5, var_2);
        //---------
        // reverse
        wp::adj_array_store(var_ray_out, var_5, var_2, adj_ray_out, adj_5, adj_2);
        wp::adj_add(var_3, var_4, adj_3, adj_4, adj_5);
        wp::adj_mul(var_1, var_img_w, adj_1, adj_img_w, adj_4);
        wp::adj_add(var_offset, var_0, adj_offset, adj_0, adj_3);
        // adj: ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                    <L 4081>
        adj_compute_ray_0(var_projection, var_fovy, var_sensorsize, var_intrinsic, var_img_w, var_img_h, var_0, var_1, var_znear, adj_projection, adj_fovy, adj_sensorsize, adj_intrinsic, adj_img_w, adj_img_h, adj_0, adj_1, adj_znear, adj_2);
        // adj: projection, fovy, sensorsize, intrinsic, img_w, img_h, xid, yid, znear            <L 4082>
        // adj: ray_out[offset + xid + yid * img_w] = render_util.compute_ray(                    <L 4081>
        // adj: xid, yid = wp.tid()                                                               <L 4080>
        // adj: def _build_rays(                                                                  <L 4067>
        return;
}



extern "C" __global__ void _finalize_body_invweight0_64175ec4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::float32> var_body_A_diag_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0_out)
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
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<2, wp::float32> var_21;
        const wp::float32 var_22 = 0.3333333333333333;
        const wp::int32 var_23 = 0;
        wp::float32* var_24;
        const wp::int32 var_25 = 1;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 2;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.3333333333333333;
        const wp::int32 var_36 = 3;
        wp::float32* var_37;
        const wp::int32 var_38 = 4;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 5;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        bool var_48;
        const wp::float32 var_49 = 1e-15;
        bool var_50;
        const wp::float32 var_51 = 1e-15;
        bool var_52;
        wp::float32 var_53;
        bool var_54;
        const wp::float32 var_55 = 1e-15;
        bool var_56;
        const wp::float32 var_57 = 1e-15;
        bool var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::vec_t<2, wp::float32> var_63;
        //---------
        // forward
        // def _finalize_body_invweight0(                                                         <L 3254>
        // worldid, bodyid = wp.tid()                                                             <L 3259>
        builtin_tid2d(var_0, var_1);
        // body_invweight0_id = worldid % body_invweight0_out.shape[0]                            <L 3260>
        var_2 = &(var_body_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_A_diag_id = worldid % body_A_diag_in.shape[0]                                     <L 3261>
        var_7 = &(var_body_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // if bodyid == 0 or body_weldid[bodyid] == 0:                                            <L 3264>
        var_14 = (var_1 == var_13);
        var_12 = var_14;
        if (!var_12) {
            var_15 = wp::address(var_body_weldid, var_1);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            var_12 = var_12 || var_17;
        }
        if (var_12) {
            // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(0.0, 0.0)                <L 3265>
            var_21 = wp::vec_t<2, wp::float32>(var_19, var_20);
            wp::array_store(var_body_invweight0_out, var_6, var_1, var_21);
            // return                                                                             <L 3266>
            return;
        }
        // inv_trans = wp.static(1.0 / 3.0) * (                                                   <L 3269>
        // body_A_diag_in[body_A_diag_id, bodyid, 0]                                              <L 3270>
        var_24 = wp::address(var_body_A_diag_in, var_11, var_1, var_23);
        // + body_A_diag_in[body_A_diag_id, bodyid, 1]                                            <L 3271>
        var_26 = wp::address(var_body_A_diag_in, var_11, var_1, var_25);
        var_28 = wp::load(var_24);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        // + body_A_diag_in[body_A_diag_id, bodyid, 2]                                            <L 3272>
        var_31 = wp::address(var_body_A_diag_in, var_11, var_1, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::add(var_27, var_33);
        var_34 = wp::mul(var_22, var_32);
        // inv_rot = wp.static(1.0 / 3.0) * (                                                     <L 3274>
        // body_A_diag_in[body_A_diag_id, bodyid, 3]                                              <L 3275>
        var_37 = wp::address(var_body_A_diag_in, var_11, var_1, var_36);
        // + body_A_diag_in[body_A_diag_id, bodyid, 4]                                            <L 3276>
        var_39 = wp::address(var_body_A_diag_in, var_11, var_1, var_38);
        var_41 = wp::load(var_37);
        var_42 = wp::load(var_39);
        var_40 = wp::add(var_41, var_42);
        // + body_A_diag_in[body_A_diag_id, bodyid, 5]                                            <L 3277>
        var_44 = wp::address(var_body_A_diag_in, var_11, var_1, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::add(var_40, var_46);
        var_47 = wp::mul(var_35, var_45);
        // if inv_trans < mujoco.mjMINVAL and inv_rot > mujoco.mjMINVAL:                          <L 3281>
        var_50 = (var_34 < var_49);
        var_48 = var_50;
        if (var_48) {
            var_52 = (var_47 > var_51);
            var_48 = var_48 && var_52;
        }
        if (var_48) {
            // inv_trans = inv_rot  # use rotation as fallback for translation                    <L 3282>
            var_53 = wp::copy(var_47);
        }
        if (!var_48) {
            // elif inv_rot < mujoco.mjMINVAL and inv_trans > mujoco.mjMINVAL:                    <L 3283>
            var_56 = (var_47 < var_55);
            var_54 = var_56;
            if (var_54) {
                var_58 = (var_34 > var_57);
                var_54 = var_54 && var_58;
            }
            if (var_54) {
                // inv_rot = inv_trans  # use translation as fallback for rotation                <L 3284>
                var_59 = wp::copy(var_34);
            }
            var_60 = wp::where(var_54, var_59, var_47);
        }
        var_61 = wp::where(var_48, var_53, var_34);
        var_62 = wp::where(var_48, var_47, var_60);
        // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(inv_trans, inv_rot)          <L 3286>
        var_63 = wp::vec_t<2, wp::float32>(var_61, var_62);
        wp::array_store(var_body_invweight0_out, var_6, var_1, var_63);
}



extern "C" __global__ void _finalize_body_invweight0_64175ec4_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::float32> var_body_A_diag_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0_out,
    wp::array_t<wp::int32> adj_body_weldid,
    wp::array_t<wp::float32> adj_body_A_diag_in,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_body_invweight0_out)
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
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        wp::vec_t<2, wp::float32> var_21;
        const wp::float32 var_22 = 0.3333333333333333;
        const wp::int32 var_23 = 0;
        wp::float32* var_24;
        const wp::int32 var_25 = 1;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 2;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.3333333333333333;
        const wp::int32 var_36 = 3;
        wp::float32* var_37;
        const wp::int32 var_38 = 4;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 5;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        bool var_48;
        const wp::float32 var_49 = 1e-15;
        bool var_50;
        const wp::float32 var_51 = 1e-15;
        bool var_52;
        wp::float32 var_53;
        bool var_54;
        const wp::float32 var_55 = 1e-15;
        bool var_56;
        const wp::float32 var_57 = 1e-15;
        bool var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::vec_t<2, wp::float32> var_63;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::shape_t adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::shape_t adj_10 = {};
        wp::int32 adj_11 = {};
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        bool adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::vec_t<2, wp::float32> adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        bool adj_48 = {};
        wp::float32 adj_49 = {};
        bool adj_50 = {};
        wp::float32 adj_51 = {};
        bool adj_52 = {};
        wp::float32 adj_53 = {};
        bool adj_54 = {};
        wp::float32 adj_55 = {};
        bool adj_56 = {};
        wp::float32 adj_57 = {};
        bool adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::vec_t<2, wp::float32> adj_63 = {};
        //---------
        // forward
        // def _finalize_body_invweight0(                                                         <L 3254>
        // worldid, bodyid = wp.tid()                                                             <L 3259>
        builtin_tid2d(var_0, var_1);
        // body_invweight0_id = worldid % body_invweight0_out.shape[0]                            <L 3260>
        var_2 = &(var_body_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_A_diag_id = worldid % body_A_diag_in.shape[0]                                     <L 3261>
        var_7 = &(var_body_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // if bodyid == 0 or body_weldid[bodyid] == 0:                                            <L 3264>
        var_14 = (var_1 == var_13);
        var_12 = var_14;
        if (!var_12) {
            var_15 = wp::address(var_body_weldid, var_1);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            var_12 = var_12 || var_17;
        }
        if (var_12) {
            // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(0.0, 0.0)                <L 3265>
            var_21 = wp::vec_t<2, wp::float32>(var_19, var_20);
            // wp::array_store(var_body_invweight0_out, var_6, var_1, var_21);
            // return                                                                             <L 3266>
            goto label0;
        }
        // inv_trans = wp.static(1.0 / 3.0) * (                                                   <L 3269>
        // body_A_diag_in[body_A_diag_id, bodyid, 0]                                              <L 3270>
        var_24 = wp::address(var_body_A_diag_in, var_11, var_1, var_23);
        // + body_A_diag_in[body_A_diag_id, bodyid, 1]                                            <L 3271>
        var_26 = wp::address(var_body_A_diag_in, var_11, var_1, var_25);
        var_28 = wp::load(var_24);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        // + body_A_diag_in[body_A_diag_id, bodyid, 2]                                            <L 3272>
        var_31 = wp::address(var_body_A_diag_in, var_11, var_1, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::add(var_27, var_33);
        var_34 = wp::mul(var_22, var_32);
        // inv_rot = wp.static(1.0 / 3.0) * (                                                     <L 3274>
        // body_A_diag_in[body_A_diag_id, bodyid, 3]                                              <L 3275>
        var_37 = wp::address(var_body_A_diag_in, var_11, var_1, var_36);
        // + body_A_diag_in[body_A_diag_id, bodyid, 4]                                            <L 3276>
        var_39 = wp::address(var_body_A_diag_in, var_11, var_1, var_38);
        var_41 = wp::load(var_37);
        var_42 = wp::load(var_39);
        var_40 = wp::add(var_41, var_42);
        // + body_A_diag_in[body_A_diag_id, bodyid, 5]                                            <L 3277>
        var_44 = wp::address(var_body_A_diag_in, var_11, var_1, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::add(var_40, var_46);
        var_47 = wp::mul(var_35, var_45);
        // if inv_trans < mujoco.mjMINVAL and inv_rot > mujoco.mjMINVAL:                          <L 3281>
        var_50 = (var_34 < var_49);
        var_48 = var_50;
        if (var_48) {
            var_52 = (var_47 > var_51);
            var_48 = var_48 && var_52;
        }
        if (var_48) {
            // inv_trans = inv_rot  # use rotation as fallback for translation                    <L 3282>
            var_53 = wp::copy(var_47);
        }
        if (!var_48) {
            // elif inv_rot < mujoco.mjMINVAL and inv_trans > mujoco.mjMINVAL:                    <L 3283>
            var_56 = (var_47 < var_55);
            var_54 = var_56;
            if (var_54) {
                var_58 = (var_34 > var_57);
                var_54 = var_54 && var_58;
            }
            if (var_54) {
                // inv_rot = inv_trans  # use translation as fallback for rotation                <L 3284>
                var_59 = wp::copy(var_34);
            }
            var_60 = wp::where(var_54, var_59, var_47);
        }
        var_61 = wp::where(var_48, var_53, var_34);
        var_62 = wp::where(var_48, var_47, var_60);
        // body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(inv_trans, inv_rot)          <L 3286>
        var_63 = wp::vec_t<2, wp::float32>(var_61, var_62);
        // wp::array_store(var_body_invweight0_out, var_6, var_1, var_63);
        //---------
        // reverse
        wp::adj_array_store(var_body_invweight0_out, var_6, var_1, var_63, adj_body_invweight0_out, adj_6, adj_1, adj_63);
        wp::adj_vec_t(var_61, var_62, adj_61, adj_62, adj_63);
        // adj: body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(inv_trans, inv_rot)     <L 3286>
        wp::adj_where(var_48, var_47, var_60, adj_48, adj_47, adj_60, adj_62);
        wp::adj_where(var_48, var_53, var_34, adj_48, adj_53, adj_34, adj_61);
        if (!var_48) {
            wp::adj_where(var_54, var_59, var_47, adj_54, adj_59, adj_47, adj_60);
            if (var_54) {
                wp::adj_copy(var_34, adj_34, adj_59);
                // adj: inv_rot = inv_trans  # use translation as fallback for rotation           <L 3284>
            }
            if (var_54) {
            }
            // adj: elif inv_rot < mujoco.mjMINVAL and inv_trans > mujoco.mjMINVAL:               <L 3283>
        }
        if (var_48) {
            wp::adj_copy(var_47, adj_47, adj_53);
            // adj: inv_trans = inv_rot  # use rotation as fallback for translation               <L 3282>
        }
        if (var_48) {
        }
        // adj: if inv_trans < mujoco.mjMINVAL and inv_rot > mujoco.mjMINVAL:                     <L 3281>
        wp::adj_mul(var_35, var_45, adj_35, adj_45, adj_47);
        wp::adj_add(var_40, var_46, adj_40, adj_44, adj_45);
        wp::adj_address(var_body_A_diag_in, var_11, var_1, var_43, adj_body_A_diag_in, adj_11, adj_1, adj_43, adj_44);
        // adj: + body_A_diag_in[body_A_diag_id, bodyid, 5]                                       <L 3277>
        wp::adj_add(var_41, var_42, adj_37, adj_39, adj_40);
        wp::adj_address(var_body_A_diag_in, var_11, var_1, var_38, adj_body_A_diag_in, adj_11, adj_1, adj_38, adj_39);
        // adj: + body_A_diag_in[body_A_diag_id, bodyid, 4]                                       <L 3276>
        wp::adj_address(var_body_A_diag_in, var_11, var_1, var_36, adj_body_A_diag_in, adj_11, adj_1, adj_36, adj_37);
        // adj: body_A_diag_in[body_A_diag_id, bodyid, 3]                                         <L 3275>
        // adj: inv_rot = wp.static(1.0 / 3.0) * (                                                <L 3274>
        wp::adj_mul(var_22, var_32, adj_22, adj_32, adj_34);
        wp::adj_add(var_27, var_33, adj_27, adj_31, adj_32);
        wp::adj_address(var_body_A_diag_in, var_11, var_1, var_30, adj_body_A_diag_in, adj_11, adj_1, adj_30, adj_31);
        // adj: + body_A_diag_in[body_A_diag_id, bodyid, 2]                                       <L 3272>
        wp::adj_add(var_28, var_29, adj_24, adj_26, adj_27);
        wp::adj_address(var_body_A_diag_in, var_11, var_1, var_25, adj_body_A_diag_in, adj_11, adj_1, adj_25, adj_26);
        // adj: + body_A_diag_in[body_A_diag_id, bodyid, 1]                                       <L 3271>
        wp::adj_address(var_body_A_diag_in, var_11, var_1, var_23, adj_body_A_diag_in, adj_11, adj_1, adj_23, adj_24);
        // adj: body_A_diag_in[body_A_diag_id, bodyid, 0]                                         <L 3270>
        // adj: inv_trans = wp.static(1.0 / 3.0) * (                                              <L 3269>
        if (var_12) {
            label0:;
            // adj: return                                                                        <L 3266>
            wp::adj_array_store(var_body_invweight0_out, var_6, var_1, var_21, adj_body_invweight0_out, adj_6, adj_1, adj_21);
            wp::adj_vec_t(var_19, var_20, adj_19, adj_20, adj_21);
            // adj: body_invweight0_out[body_invweight0_id, bodyid] = wp.vec2(0.0, 0.0)           <L 3265>
        }
        if (!var_12) {
            wp::adj_address(var_body_weldid, var_1, adj_body_weldid, adj_1, adj_15);
        }
        // adj: if bodyid == 0 or body_weldid[bodyid] == 0:                                       <L 3264>
        wp::adj_mod(var_0, var_9, adj_0, adj_9, adj_11);
        adj_body_A_diag_in.shape = adj_7;
        // adj: body_A_diag_id = worldid % body_A_diag_in.shape[0]                                <L 3261>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_body_invweight0_out.shape = adj_2;
        // adj: body_invweight0_id = worldid % body_invweight0_out.shape[0]                       <L 3260>
        // adj: worldid, bodyid = wp.tid()                                                        <L 3259>
        // adj: def _finalize_body_invweight0(                                                    <L 3254>
        return;
}



extern "C" __global__ void _compute_body_jac_row_6eb5882a_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_body_jac_row_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::range_t var_1;
        wp::int32 var_2;
        const wp::float32 var_3 = 0.0;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::int32* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::vec_t<6, wp::float32>* var_34;
        wp::vec_t<6, wp::float32> var_35;
        wp::vec_t<6, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        const wp::int32 var_39 = 3;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        const wp::int32 var_46 = 0;
        wp::float32 var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        bool var_50;
        const wp::int32 var_51 = 1;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 2;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 3;
        bool var_62;
        const wp::int32 var_63 = 0;
        wp::float32 var_64;
        const wp::int32 var_65 = 4;
        bool var_66;
        const wp::int32 var_67 = 1;
        wp::float32 var_68;
        const wp::int32 var_69 = 2;
        wp::float32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        //---------
        // forward
        // def _compute_body_jac_row(                                                             <L 3177>
        // worldid = wp.tid()                                                                     <L 3191>
        var_0 = builtin_tid1d();
        // for i in range(nv):                                                                    <L 3193>
        var_1 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_1) == 0) goto end_for_0;
            var_2 = wp::iter_next(var_1);
            // body_jac_row_out[worldid, i] = 0.0                                                 <L 3194>
            wp::array_store(var_body_jac_row_out, var_0, var_2, var_3);
            goto start_for_0;
        end_for_0:;
        // bodyid = bodyid_target                                                                 <L 3196>
        var_4 = wp::copy(var_bodyid_target);
        // while bodyid > 0 and body_dofnum[bodyid] == 0:                                         <L 3197>
        start_while_2:;
        var_7 = (var_4 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = wp::address(var_body_dofnum, var_4);
            var_11 = wp::load(var_8);
            var_10 = (var_11 == var_9);
            var_5 = var_5 && var_10;
        }
        if ((var_5) == false) goto end_while_2;
            // bodyid = body_parentid[bodyid]                                                     <L 3198>
            var_12 = wp::address(var_body_parentid, var_4);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            wp::assign(var_4, var_13);
        goto start_while_2;
        end_while_2:;
        // if bodyid == 0:                                                                        <L 3200>
        var_16 = (var_4 == var_15);
        if (var_16) {
            // return                                                                             <L 3201>
            return;
        }
        // point = xipos_in[worldid, bodyid_target]                                               <L 3204>
        var_17 = wp::address(var_xipos_in, var_0, var_bodyid_target);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // offset = point - subtree_com_in[worldid, body_rootid[bodyid_target]]                   <L 3205>
        var_20 = wp::address(var_body_rootid, var_bodyid_target);
        var_22 = wp::load(var_20);
        var_21 = wp::address(var_subtree_com_in, var_0, var_22);
        var_24 = wp::load(var_21);
        var_23 = wp::sub(var_18, var_24);
        // dofid = body_dofadr[bodyid] + body_dofnum[bodyid] - 1                                  <L 3208>
        var_25 = wp::address(var_body_dofadr, var_4);
        var_26 = wp::address(var_body_dofnum, var_4);
        var_28 = wp::load(var_25);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        var_31 = wp::sub(var_27, var_30);
        // while dofid >= 0:                                                                      <L 3211>
        start_while_5:;
        var_33 = (var_31 >= var_32);
        if ((var_33) == false) goto end_while_5;
            // cdof = cdof_in[worldid, dofid]                                                     <L 3212>
            var_34 = wp::address(var_cdof_in, var_0, var_31);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // cdof_ang = wp.spatial_top(cdof)                                                    <L 3213>
            var_37 = wp::spatial_top(var_35);
            // cdof_lin = wp.spatial_bottom(cdof)                                                 <L 3214>
            var_38 = wp::spatial_bottom(var_35);
            // if row_idx < 3:                                                                    <L 3216>
            var_40 = (var_row_idx < var_39);
            if (var_40) {
                // tmp = wp.cross(cdof_ang, offset)                                               <L 3217>
                var_41 = wp::cross(var_37, var_23);
                // if row_idx == 0:                                                               <L 3218>
                var_43 = (var_row_idx == var_42);
                if (var_43) {
                    // body_jac_row_out[worldid, dofid] = cdof_lin[0] + tmp[0]                    <L 3219>
                    var_45 = wp::extract(var_38, var_44);
                    var_47 = wp::extract(var_41, var_46);
                    var_48 = wp::add(var_45, var_47);
                    wp::array_store(var_body_jac_row_out, var_0, var_31, var_48);
                }
                if (!var_43) {
                    // elif row_idx == 1:                                                         <L 3220>
                    var_50 = (var_row_idx == var_49);
                    if (var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[1] + tmp[1]                <L 3221>
                        var_52 = wp::extract(var_38, var_51);
                        var_54 = wp::extract(var_41, var_53);
                        var_55 = wp::add(var_52, var_54);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_55);
                    }
                    if (!var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[2] + tmp[2]                <L 3223>
                        var_57 = wp::extract(var_38, var_56);
                        var_59 = wp::extract(var_41, var_58);
                        var_60 = wp::add(var_57, var_59);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_60);
                    }
                }
            }
            if (!var_40) {
                // if row_idx == 3:                                                               <L 3225>
                var_62 = (var_row_idx == var_61);
                if (var_62) {
                    // body_jac_row_out[worldid, dofid] = cdof_ang[0]                             <L 3226>
                    var_64 = wp::extract(var_37, var_63);
                    wp::array_store(var_body_jac_row_out, var_0, var_31, var_64);
                }
                if (!var_62) {
                    // elif row_idx == 4:                                                         <L 3227>
                    var_66 = (var_row_idx == var_65);
                    if (var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[1]                         <L 3228>
                        var_68 = wp::extract(var_37, var_67);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_68);
                    }
                    if (!var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[2]                         <L 3230>
                        var_70 = wp::extract(var_37, var_69);
                        wp::array_store(var_body_jac_row_out, var_0, var_31, var_70);
                    }
                }
            }
            // dofid = dof_parentid[dofid]                                                        <L 3232>
            var_71 = wp::address(var_dof_parentid, var_31);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            wp::assign(var_31, var_72);
        goto start_while_5;
        end_while_5:;
}



extern "C" __global__ void _compute_body_jac_row_6eb5882a_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_body_jac_row_out,
    wp::int32 adj_nv,
    wp::int32 adj_bodyid_target,
    wp::int32 adj_row_idx,
    wp::array_t<wp::int32> adj_body_parentid,
    wp::array_t<wp::int32> adj_body_rootid,
    wp::array_t<wp::int32> adj_body_dofadr,
    wp::array_t<wp::int32> adj_body_dofnum,
    wp::array_t<wp::int32> adj_dof_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_xipos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_cdof_in,
    wp::array_t<wp::float32> adj_body_jac_row_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::range_t var_1;
        wp::int32 var_2;
        const wp::float32 var_3 = 0.0;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::int32* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::vec_t<6, wp::float32>* var_34;
        wp::vec_t<6, wp::float32> var_35;
        wp::vec_t<6, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        const wp::int32 var_39 = 3;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        const wp::int32 var_46 = 0;
        wp::float32 var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        bool var_50;
        const wp::int32 var_51 = 1;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 2;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 3;
        bool var_62;
        const wp::int32 var_63 = 0;
        wp::float32 var_64;
        const wp::int32 var_65 = 4;
        bool var_66;
        const wp::int32 var_67 = 1;
        wp::float32 var_68;
        const wp::int32 var_69 = 2;
        wp::float32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::range_t adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        bool adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::int32 adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        bool adj_33 = {};
        wp::vec_t<6, wp::float32> adj_34 = {};
        wp::vec_t<6, wp::float32> adj_35 = {};
        wp::vec_t<6, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::int32 adj_39 = {};
        bool adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        bool adj_50 = {};
        wp::int32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::int32 adj_61 = {};
        bool adj_62 = {};
        wp::int32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::int32 adj_65 = {};
        bool adj_66 = {};
        wp::int32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        //---------
        // forward
        // def _compute_body_jac_row(                                                             <L 3177>
        // worldid = wp.tid()                                                                     <L 3191>
        var_0 = builtin_tid1d();
        // for i in range(nv):                                                                    <L 3193>
        var_1 = wp::range(var_nv);
        // bodyid = bodyid_target                                                                 <L 3196>
        var_4 = wp::copy(var_bodyid_target);
        // while bodyid > 0 and body_dofnum[bodyid] == 0:                                         <L 3197>
        // if bodyid == 0:                                                                        <L 3200>
        var_16 = (var_4 == var_15);
        if (var_16) {
            // return                                                                             <L 3201>
            goto label4;
        }
        // point = xipos_in[worldid, bodyid_target]                                               <L 3204>
        var_17 = wp::address(var_xipos_in, var_0, var_bodyid_target);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // offset = point - subtree_com_in[worldid, body_rootid[bodyid_target]]                   <L 3205>
        var_20 = wp::address(var_body_rootid, var_bodyid_target);
        var_22 = wp::load(var_20);
        var_21 = wp::address(var_subtree_com_in, var_0, var_22);
        var_24 = wp::load(var_21);
        var_23 = wp::sub(var_18, var_24);
        // dofid = body_dofadr[bodyid] + body_dofnum[bodyid] - 1                                  <L 3208>
        var_25 = wp::address(var_body_dofadr, var_4);
        var_26 = wp::address(var_body_dofnum, var_4);
        var_28 = wp::load(var_25);
        var_29 = wp::load(var_26);
        var_27 = wp::add(var_28, var_29);
        var_31 = wp::sub(var_27, var_30);
        // while dofid >= 0:                                                                      <L 3211>
        //---------
        // reverse
        start_while_5:;
        var_33 = (var_31 >= var_32);
        if ((var_33) == false) goto end_while_5;
        adj_34 = {};
        adj_35 = {};
        adj_36 = {};
        adj_37 = {};
        adj_38 = {};
        adj_39 = {};
        adj_40 = {};
        adj_41 = {};
        adj_42 = {};
        adj_43 = {};
        adj_44 = {};
        adj_45 = {};
        adj_46 = {};
        adj_47 = {};
        adj_48 = {};
        adj_49 = {};
        adj_50 = {};
        adj_51 = {};
        adj_52 = {};
        adj_53 = {};
        adj_54 = {};
        adj_55 = {};
        adj_56 = {};
        adj_57 = {};
        adj_58 = {};
        adj_59 = {};
        adj_60 = {};
        adj_61 = {};
        adj_62 = {};
        adj_63 = {};
        adj_64 = {};
        adj_65 = {};
        adj_66 = {};
        adj_67 = {};
        adj_68 = {};
        adj_69 = {};
        adj_70 = {};
        adj_71 = {};
        adj_72 = {};
        adj_73 = {};
            // cdof = cdof_in[worldid, dofid]                                                     <L 3212>
            var_34 = wp::address(var_cdof_in, var_0, var_31);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // cdof_ang = wp.spatial_top(cdof)                                                    <L 3213>
            var_37 = wp::spatial_top(var_35);
            // cdof_lin = wp.spatial_bottom(cdof)                                                 <L 3214>
            var_38 = wp::spatial_bottom(var_35);
            // if row_idx < 3:                                                                    <L 3216>
            var_40 = (var_row_idx < var_39);
            if (var_40) {
                // tmp = wp.cross(cdof_ang, offset)                                               <L 3217>
                var_41 = wp::cross(var_37, var_23);
                // if row_idx == 0:                                                               <L 3218>
                var_43 = (var_row_idx == var_42);
                if (var_43) {
                    // body_jac_row_out[worldid, dofid] = cdof_lin[0] + tmp[0]                    <L 3219>
                    var_45 = wp::extract(var_38, var_44);
                    var_47 = wp::extract(var_41, var_46);
                    var_48 = wp::add(var_45, var_47);
                    // wp::array_store(var_body_jac_row_out, var_0, var_31, var_48);
                }
                if (!var_43) {
                    // elif row_idx == 1:                                                         <L 3220>
                    var_50 = (var_row_idx == var_49);
                    if (var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[1] + tmp[1]                <L 3221>
                        var_52 = wp::extract(var_38, var_51);
                        var_54 = wp::extract(var_41, var_53);
                        var_55 = wp::add(var_52, var_54);
                        // wp::array_store(var_body_jac_row_out, var_0, var_31, var_55);
                    }
                    if (!var_50) {
                        // body_jac_row_out[worldid, dofid] = cdof_lin[2] + tmp[2]                <L 3223>
                        var_57 = wp::extract(var_38, var_56);
                        var_59 = wp::extract(var_41, var_58);
                        var_60 = wp::add(var_57, var_59);
                        // wp::array_store(var_body_jac_row_out, var_0, var_31, var_60);
                    }
                }
            }
            if (!var_40) {
                // if row_idx == 3:                                                               <L 3225>
                var_62 = (var_row_idx == var_61);
                if (var_62) {
                    // body_jac_row_out[worldid, dofid] = cdof_ang[0]                             <L 3226>
                    var_64 = wp::extract(var_37, var_63);
                    // wp::array_store(var_body_jac_row_out, var_0, var_31, var_64);
                }
                if (!var_62) {
                    // elif row_idx == 4:                                                         <L 3227>
                    var_66 = (var_row_idx == var_65);
                    if (var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[1]                         <L 3228>
                        var_68 = wp::extract(var_37, var_67);
                        // wp::array_store(var_body_jac_row_out, var_0, var_31, var_68);
                    }
                    if (!var_66) {
                        // body_jac_row_out[worldid, dofid] = cdof_ang[2]                         <L 3230>
                        var_70 = wp::extract(var_37, var_69);
                        // wp::array_store(var_body_jac_row_out, var_0, var_31, var_70);
                    }
                }
            }
            // dofid = dof_parentid[dofid]                                                        <L 3232>
            var_71 = wp::address(var_dof_parentid, var_31);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            wp::assign(var_31, var_72);
            wp::adj_assign(var_31, var_72, adj_31, adj_72);
            wp::adj_copy(var_73, adj_71, adj_72);
            wp::adj_address(var_dof_parentid, var_31, adj_dof_parentid, adj_31, adj_71);
            // adj: dofid = dof_parentid[dofid]                                                   <L 3232>
            if (!var_40) {
                if (!var_62) {
                    if (!var_66) {
                        wp::adj_array_store(var_body_jac_row_out, var_0, var_31, var_70, adj_body_jac_row_out, adj_0, adj_31, adj_70);
                        wp::adj_extract(var_37, var_69, adj_37, adj_69, adj_70);
                        // adj: body_jac_row_out[worldid, dofid] = cdof_ang[2]                    <L 3230>
                    }
                    if (var_66) {
                        wp::adj_array_store(var_body_jac_row_out, var_0, var_31, var_68, adj_body_jac_row_out, adj_0, adj_31, adj_68);
                        wp::adj_extract(var_37, var_67, adj_37, adj_67, adj_68);
                        // adj: body_jac_row_out[worldid, dofid] = cdof_ang[1]                    <L 3228>
                    }
                    // adj: elif row_idx == 4:                                                    <L 3227>
                }
                if (var_62) {
                    wp::adj_array_store(var_body_jac_row_out, var_0, var_31, var_64, adj_body_jac_row_out, adj_0, adj_31, adj_64);
                    wp::adj_extract(var_37, var_63, adj_37, adj_63, adj_64);
                    // adj: body_jac_row_out[worldid, dofid] = cdof_ang[0]                        <L 3226>
                }
                // adj: if row_idx == 3:                                                          <L 3225>
            }
            if (var_40) {
                if (!var_43) {
                    if (!var_50) {
                        wp::adj_array_store(var_body_jac_row_out, var_0, var_31, var_60, adj_body_jac_row_out, adj_0, adj_31, adj_60);
                        wp::adj_add(var_57, var_59, adj_57, adj_59, adj_60);
                        wp::adj_extract(var_41, var_58, adj_41, adj_58, adj_59);
                        wp::adj_extract(var_38, var_56, adj_38, adj_56, adj_57);
                        // adj: body_jac_row_out[worldid, dofid] = cdof_lin[2] + tmp[2]           <L 3223>
                    }
                    if (var_50) {
                        wp::adj_array_store(var_body_jac_row_out, var_0, var_31, var_55, adj_body_jac_row_out, adj_0, adj_31, adj_55);
                        wp::adj_add(var_52, var_54, adj_52, adj_54, adj_55);
                        wp::adj_extract(var_41, var_53, adj_41, adj_53, adj_54);
                        wp::adj_extract(var_38, var_51, adj_38, adj_51, adj_52);
                        // adj: body_jac_row_out[worldid, dofid] = cdof_lin[1] + tmp[1]           <L 3221>
                    }
                    // adj: elif row_idx == 1:                                                    <L 3220>
                }
                if (var_43) {
                    wp::adj_array_store(var_body_jac_row_out, var_0, var_31, var_48, adj_body_jac_row_out, adj_0, adj_31, adj_48);
                    wp::adj_add(var_45, var_47, adj_45, adj_47, adj_48);
                    wp::adj_extract(var_41, var_46, adj_41, adj_46, adj_47);
                    wp::adj_extract(var_38, var_44, adj_38, adj_44, adj_45);
                    // adj: body_jac_row_out[worldid, dofid] = cdof_lin[0] + tmp[0]               <L 3219>
                }
                // adj: if row_idx == 0:                                                          <L 3218>
                wp::adj_cross(var_37, var_23, adj_37, adj_23, adj_41);
                // adj: tmp = wp.cross(cdof_ang, offset)                                          <L 3217>
            }
            // adj: if row_idx < 3:                                                               <L 3216>
            wp::adj_spatial_bottom(var_35, adj_35, adj_38);
            // adj: cdof_lin = wp.spatial_bottom(cdof)                                            <L 3214>
            wp::adj_spatial_top(var_35, adj_35, adj_37);
            // adj: cdof_ang = wp.spatial_top(cdof)                                               <L 3213>
            wp::adj_copy(var_36, adj_34, adj_35);
            wp::adj_address(var_cdof_in, var_0, var_31, adj_cdof_in, adj_0, adj_31, adj_34);
            // adj: cdof = cdof_in[worldid, dofid]                                                <L 3212>
        goto start_while_5;
        end_while_5:;
        // adj: while dofid >= 0:                                                                 <L 3211>
        wp::adj_sub(var_27, var_30, adj_27, adj_30, adj_31);
        wp::adj_add(var_28, var_29, adj_25, adj_26, adj_27);
        wp::adj_address(var_body_dofnum, var_4, adj_body_dofnum, adj_4, adj_26);
        wp::adj_address(var_body_dofadr, var_4, adj_body_dofadr, adj_4, adj_25);
        // adj: dofid = body_dofadr[bodyid] + body_dofnum[bodyid] - 1                             <L 3208>
        wp::adj_sub(var_18, var_24, adj_18, adj_21, adj_23);
        wp::adj_address(var_subtree_com_in, var_0, var_22, adj_subtree_com_in, adj_0, adj_20, adj_21);
        wp::adj_address(var_body_rootid, var_bodyid_target, adj_body_rootid, adj_bodyid_target, adj_20);
        // adj: offset = point - subtree_com_in[worldid, body_rootid[bodyid_target]]              <L 3205>
        wp::adj_copy(var_19, adj_17, adj_18);
        wp::adj_address(var_xipos_in, var_0, var_bodyid_target, adj_xipos_in, adj_0, adj_bodyid_target, adj_17);
        // adj: point = xipos_in[worldid, bodyid_target]                                          <L 3204>
        if (var_16) {
            label4:;
            // adj: return                                                                        <L 3201>
        }
        // adj: if bodyid == 0:                                                                   <L 3200>
        start_while_2:;
        var_7 = (var_4 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = wp::address(var_body_dofnum, var_4);
            var_11 = wp::load(var_8);
            var_10 = (var_11 == var_9);
            var_5 = var_5 && var_10;
        }
        if ((var_5) == false) goto end_while_2;
        adj_12 = {};
        adj_13 = {};
        adj_14 = {};
            // bodyid = body_parentid[bodyid]                                                     <L 3198>
            var_12 = wp::address(var_body_parentid, var_4);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            wp::assign(var_4, var_13);
            wp::adj_assign(var_4, var_13, adj_4, adj_13);
            wp::adj_copy(var_14, adj_12, adj_13);
            wp::adj_address(var_body_parentid, var_4, adj_body_parentid, adj_4, adj_12);
            // adj: bodyid = body_parentid[bodyid]                                                <L 3198>
        goto start_while_2;
        end_while_2:;
        // adj: while bodyid > 0 and body_dofnum[bodyid] == 0:                                    <L 3197>
        wp::adj_copy(var_bodyid_target, adj_bodyid_target, adj_4);
        // adj: bodyid = bodyid_target                                                            <L 3196>
        var_1 = wp::iter_reverse(var_1);
        start_for_0:;
            if (iter_cmp(var_1) == 0) goto end_for_0;
            var_2 = wp::iter_next(var_1);
        	adj_3 = {};
            // body_jac_row_out[worldid, i] = 0.0                                                 <L 3194>
            // wp::array_store(var_body_jac_row_out, var_0, var_2, var_3);
            wp::adj_array_store(var_body_jac_row_out, var_0, var_2, var_3, adj_body_jac_row_out, adj_0, adj_2, adj_3);
            // adj: body_jac_row_out[worldid, i] = 0.0                                            <L 3194>
        	goto start_for_0;
        end_for_0:;
        // adj: for i in range(nv):                                                               <L 3193>
        // adj: worldid = wp.tid()                                                                <L 3191>
        // adj: def _compute_body_jac_row(                                                        <L 3177>
        return;
}



extern "C" __global__ void _extract_dof_A_diag_917a07a1_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_dof_A_diag_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        //---------
        // forward
        // def _extract_dof_A_diag(                                                               <L 3122>
        // worldid = wp.tid()                                                                     <L 3127>
        var_0 = builtin_tid1d();
        // dof_A_diag_id = worldid % dof_A_diag_out.shape[0]                                      <L 3128>
        var_1 = &(var_dof_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dof_A_diag_out[dof_A_diag_id, dofid] = result_vec_in[worldid, dofid]                   <L 3129>
        var_6 = wp::address(var_result_vec_in, var_0, var_dofid);
        var_7 = wp::load(var_6);
        wp::array_store(var_dof_A_diag_out, var_5, var_dofid, var_7);
}



extern "C" __global__ void _extract_dof_A_diag_917a07a1_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_dof_A_diag_out,
    wp::int32 adj_dofid,
    wp::array_t<wp::float32> adj_result_vec_in,
    wp::array_t<wp::float32> adj_dof_A_diag_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::shape_t adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        //---------
        // forward
        // def _extract_dof_A_diag(                                                               <L 3122>
        // worldid = wp.tid()                                                                     <L 3127>
        var_0 = builtin_tid1d();
        // dof_A_diag_id = worldid % dof_A_diag_out.shape[0]                                      <L 3128>
        var_1 = &(var_dof_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dof_A_diag_out[dof_A_diag_id, dofid] = result_vec_in[worldid, dofid]                   <L 3129>
        var_6 = wp::address(var_result_vec_in, var_0, var_dofid);
        var_7 = wp::load(var_6);
        // wp::array_store(var_dof_A_diag_out, var_5, var_dofid, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_dof_A_diag_out, var_5, var_dofid, var_7, adj_dof_A_diag_out, adj_5, adj_dofid, adj_6);
        wp::adj_address(var_result_vec_in, var_0, var_dofid, adj_result_vec_in, adj_0, adj_dofid, adj_6);
        // adj: dof_A_diag_out[dof_A_diag_id, dofid] = result_vec_in[worldid, dofid]              <L 3129>
        wp::adj_mod(var_0, var_3, adj_0, adj_3, adj_5);
        adj_dof_A_diag_out.shape = adj_1;
        // adj: dof_A_diag_id = worldid % dof_A_diag_out.shape[0]                                 <L 3128>
        // adj: worldid = wp.tid()                                                                <L 3127>
        // adj: def _extract_dof_A_diag(                                                          <L 3122>
        return;
}



extern "C" __global__ void _finalize_dof_invweight0_966ce0d5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_dof_A_diag_in,
    wp::array_t<wp::float32> var_dof_invweight0_out)
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
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 3;
        wp::int32 var_26;
        bool var_27;
        const wp::float32 var_28 = 0.3333333333333333;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::float32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.3333333333333333;
        const wp::int32 var_45 = 3;
        wp::int32 var_46;
        wp::float32* var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        const wp::int32 var_54 = 5;
        wp::int32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        bool var_64;
        const wp::float32 var_65 = 0.3333333333333333;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 2;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32* var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        //---------
        // forward
        // def _finalize_dof_invweight0(                                                          <L 3133>
        // worldid, dofid = wp.tid()                                                              <L 3140>
        builtin_tid2d(var_0, var_1);
        // dof_invweight0_id = worldid % dof_invweight0_out.shape[0]                              <L 3141>
        var_2 = &(var_dof_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // dof_A_diag_id = worldid % dof_A_diag_in.shape[0]                                       <L 3142>
        var_7 = &(var_dof_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // jntid = dof_jntid[dofid]                                                               <L 3144>
        var_12 = wp::address(var_dof_jntid, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // jtype = jnt_type[jntid]                                                                <L 3145>
        var_15 = wp::address(var_jnt_type, var_13);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // dofadr = jnt_dofadr[jntid]                                                             <L 3146>
        var_18 = wp::address(var_jnt_dofadr, var_13);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if jtype == int(types.JointType.FREE.value):                                           <L 3148>
        var_23 = wp::int(var_22);
        var_24 = (var_16 == var_23);
        if (var_24) {
            // if dofid < dofadr + 3:                                                             <L 3150>
            var_26 = wp::add(var_19, var_25);
            var_27 = (var_1 < var_26);
            if (var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 3151>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 3152>
                var_30 = wp::add(var_19, var_29);
                var_31 = wp::address(var_dof_A_diag_in, var_11, var_30);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 3153>
                var_33 = wp::add(var_19, var_32);
                var_34 = wp::address(var_dof_A_diag_in, var_11, var_33);
                var_36 = wp::load(var_31);
                var_37 = wp::load(var_34);
                var_35 = wp::add(var_36, var_37);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 3154>
                var_39 = wp::add(var_19, var_38);
                var_40 = wp::address(var_dof_A_diag_in, var_11, var_39);
                var_42 = wp::load(var_40);
                var_41 = wp::add(var_35, var_42);
                var_43 = wp::mul(var_28, var_41);
            }
            if (!var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 3157>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 3]                                       <L 3158>
                var_46 = wp::add(var_19, var_45);
                var_47 = wp::address(var_dof_A_diag_in, var_11, var_46);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 4]                                     <L 3159>
                var_49 = wp::add(var_19, var_48);
                var_50 = wp::address(var_dof_A_diag_in, var_11, var_49);
                var_52 = wp::load(var_47);
                var_53 = wp::load(var_50);
                var_51 = wp::add(var_52, var_53);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 5]                                     <L 3160>
                var_55 = wp::add(var_19, var_54);
                var_56 = wp::address(var_dof_A_diag_in, var_11, var_55);
                var_58 = wp::load(var_56);
                var_57 = wp::add(var_51, var_58);
                var_59 = wp::mul(var_44, var_57);
            }
            var_60 = wp::where(var_27, var_43, var_59);
            // dof_invweight0_out[dof_invweight0_id, dofid] = avg                                 <L 3162>
            wp::array_store(var_dof_invweight0_out, var_6, var_1, var_60);
        }
        if (!var_24) {
            // elif jtype == int(types.JointType.BALL.value):                                     <L 3163>
            var_63 = wp::int(var_62);
            var_64 = (var_16 == var_63);
            if (var_64) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 3165>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 3166>
                var_67 = wp::add(var_19, var_66);
                var_68 = wp::address(var_dof_A_diag_in, var_11, var_67);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 3167>
                var_70 = wp::add(var_19, var_69);
                var_71 = wp::address(var_dof_A_diag_in, var_11, var_70);
                var_73 = wp::load(var_68);
                var_74 = wp::load(var_71);
                var_72 = wp::add(var_73, var_74);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 3168>
                var_76 = wp::add(var_19, var_75);
                var_77 = wp::address(var_dof_A_diag_in, var_11, var_76);
                var_79 = wp::load(var_77);
                var_78 = wp::add(var_72, var_79);
                var_80 = wp::mul(var_65, var_78);
                // dof_invweight0_out[dof_invweight0_id, dofid] = avg                             <L 3170>
                wp::array_store(var_dof_invweight0_out, var_6, var_1, var_80);
            }
            if (!var_64) {
                // dof_invweight0_out[dof_invweight0_id, dofid] = dof_A_diag_in[dof_A_diag_id, dofid]       <L 3173>
                var_81 = wp::address(var_dof_A_diag_in, var_11, var_1);
                var_82 = wp::load(var_81);
                wp::array_store(var_dof_invweight0_out, var_6, var_1, var_82);
            }
        }
        var_83 = wp::where(var_24, var_60, var_80);
}



extern "C" __global__ void _finalize_dof_invweight0_966ce0d5_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_dof_A_diag_in,
    wp::array_t<wp::float32> var_dof_invweight0_out,
    wp::array_t<wp::int32> adj_dof_jntid,
    wp::array_t<wp::int32> adj_jnt_type,
    wp::array_t<wp::int32> adj_jnt_dofadr,
    wp::array_t<wp::float32> adj_dof_A_diag_in,
    wp::array_t<wp::float32> adj_dof_invweight0_out)
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
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 3;
        wp::int32 var_26;
        bool var_27;
        const wp::float32 var_28 = 0.3333333333333333;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::float32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.3333333333333333;
        const wp::int32 var_45 = 3;
        wp::int32 var_46;
        wp::float32* var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        const wp::int32 var_54 = 5;
        wp::int32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        bool var_64;
        const wp::float32 var_65 = 0.3333333333333333;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 2;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32* var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::shape_t adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::shape_t adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        bool adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        bool adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        bool adj_64 = {};
        wp::float32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::float32 adj_83 = {};
        //---------
        // forward
        // def _finalize_dof_invweight0(                                                          <L 3133>
        // worldid, dofid = wp.tid()                                                              <L 3140>
        builtin_tid2d(var_0, var_1);
        // dof_invweight0_id = worldid % dof_invweight0_out.shape[0]                              <L 3141>
        var_2 = &(var_dof_invweight0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // dof_A_diag_id = worldid % dof_A_diag_in.shape[0]                                       <L 3142>
        var_7 = &(var_dof_A_diag_in.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // jntid = dof_jntid[dofid]                                                               <L 3144>
        var_12 = wp::address(var_dof_jntid, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // jtype = jnt_type[jntid]                                                                <L 3145>
        var_15 = wp::address(var_jnt_type, var_13);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // dofadr = jnt_dofadr[jntid]                                                             <L 3146>
        var_18 = wp::address(var_jnt_dofadr, var_13);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if jtype == int(types.JointType.FREE.value):                                           <L 3148>
        var_23 = wp::int(var_22);
        var_24 = (var_16 == var_23);
        if (var_24) {
            // if dofid < dofadr + 3:                                                             <L 3150>
            var_26 = wp::add(var_19, var_25);
            var_27 = (var_1 < var_26);
            if (var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 3151>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 3152>
                var_30 = wp::add(var_19, var_29);
                var_31 = wp::address(var_dof_A_diag_in, var_11, var_30);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 3153>
                var_33 = wp::add(var_19, var_32);
                var_34 = wp::address(var_dof_A_diag_in, var_11, var_33);
                var_36 = wp::load(var_31);
                var_37 = wp::load(var_34);
                var_35 = wp::add(var_36, var_37);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 3154>
                var_39 = wp::add(var_19, var_38);
                var_40 = wp::address(var_dof_A_diag_in, var_11, var_39);
                var_42 = wp::load(var_40);
                var_41 = wp::add(var_35, var_42);
                var_43 = wp::mul(var_28, var_41);
            }
            if (!var_27) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 3157>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 3]                                       <L 3158>
                var_46 = wp::add(var_19, var_45);
                var_47 = wp::address(var_dof_A_diag_in, var_11, var_46);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 4]                                     <L 3159>
                var_49 = wp::add(var_19, var_48);
                var_50 = wp::address(var_dof_A_diag_in, var_11, var_49);
                var_52 = wp::load(var_47);
                var_53 = wp::load(var_50);
                var_51 = wp::add(var_52, var_53);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 5]                                     <L 3160>
                var_55 = wp::add(var_19, var_54);
                var_56 = wp::address(var_dof_A_diag_in, var_11, var_55);
                var_58 = wp::load(var_56);
                var_57 = wp::add(var_51, var_58);
                var_59 = wp::mul(var_44, var_57);
            }
            var_60 = wp::where(var_27, var_43, var_59);
            // dof_invweight0_out[dof_invweight0_id, dofid] = avg                                 <L 3162>
            // wp::array_store(var_dof_invweight0_out, var_6, var_1, var_60);
        }
        if (!var_24) {
            // elif jtype == int(types.JointType.BALL.value):                                     <L 3163>
            var_63 = wp::int(var_62);
            var_64 = (var_16 == var_63);
            if (var_64) {
                // avg = wp.static(1.0 / 3.0) * (                                                 <L 3165>
                // dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                       <L 3166>
                var_67 = wp::add(var_19, var_66);
                var_68 = wp::address(var_dof_A_diag_in, var_11, var_67);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                     <L 3167>
                var_70 = wp::add(var_19, var_69);
                var_71 = wp::address(var_dof_A_diag_in, var_11, var_70);
                var_73 = wp::load(var_68);
                var_74 = wp::load(var_71);
                var_72 = wp::add(var_73, var_74);
                // + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                     <L 3168>
                var_76 = wp::add(var_19, var_75);
                var_77 = wp::address(var_dof_A_diag_in, var_11, var_76);
                var_79 = wp::load(var_77);
                var_78 = wp::add(var_72, var_79);
                var_80 = wp::mul(var_65, var_78);
                // dof_invweight0_out[dof_invweight0_id, dofid] = avg                             <L 3170>
                // wp::array_store(var_dof_invweight0_out, var_6, var_1, var_80);
            }
            if (!var_64) {
                // dof_invweight0_out[dof_invweight0_id, dofid] = dof_A_diag_in[dof_A_diag_id, dofid]       <L 3173>
                var_81 = wp::address(var_dof_A_diag_in, var_11, var_1);
                var_82 = wp::load(var_81);
                // wp::array_store(var_dof_invweight0_out, var_6, var_1, var_82);
            }
        }
        var_83 = wp::where(var_24, var_60, var_80);
        //---------
        // reverse
        wp::adj_where(var_24, var_60, var_80, adj_24, adj_60, adj_80, adj_83);
        if (!var_24) {
            if (!var_64) {
                wp::adj_array_store(var_dof_invweight0_out, var_6, var_1, var_82, adj_dof_invweight0_out, adj_6, adj_1, adj_81);
                wp::adj_address(var_dof_A_diag_in, var_11, var_1, adj_dof_A_diag_in, adj_11, adj_1, adj_81);
                // adj: dof_invweight0_out[dof_invweight0_id, dofid] = dof_A_diag_in[dof_A_diag_id, dofid]  <L 3173>
            }
            if (var_64) {
                wp::adj_array_store(var_dof_invweight0_out, var_6, var_1, var_80, adj_dof_invweight0_out, adj_6, adj_1, adj_80);
                // adj: dof_invweight0_out[dof_invweight0_id, dofid] = avg                        <L 3170>
                wp::adj_mul(var_65, var_78, adj_65, adj_78, adj_80);
                wp::adj_add(var_72, var_79, adj_72, adj_77, adj_78);
                wp::adj_address(var_dof_A_diag_in, var_11, var_76, adj_dof_A_diag_in, adj_11, adj_76, adj_77);
                wp::adj_add(var_19, var_75, adj_19, adj_75, adj_76);
                // adj: + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                <L 3168>
                wp::adj_add(var_73, var_74, adj_68, adj_71, adj_72);
                wp::adj_address(var_dof_A_diag_in, var_11, var_70, adj_dof_A_diag_in, adj_11, adj_70, adj_71);
                wp::adj_add(var_19, var_69, adj_19, adj_69, adj_70);
                // adj: + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                <L 3167>
                wp::adj_address(var_dof_A_diag_in, var_11, var_67, adj_dof_A_diag_in, adj_11, adj_67, adj_68);
                wp::adj_add(var_19, var_66, adj_19, adj_66, adj_67);
                // adj: dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                  <L 3166>
                // adj: avg = wp.static(1.0 / 3.0) * (                                            <L 3165>
            }
            // adj: elif jtype == int(types.JointType.BALL.value):                                <L 3163>
        }
        if (var_24) {
            wp::adj_array_store(var_dof_invweight0_out, var_6, var_1, var_60, adj_dof_invweight0_out, adj_6, adj_1, adj_60);
            // adj: dof_invweight0_out[dof_invweight0_id, dofid] = avg                            <L 3162>
            wp::adj_where(var_27, var_43, var_59, adj_27, adj_43, adj_59, adj_60);
            if (!var_27) {
                wp::adj_mul(var_44, var_57, adj_44, adj_57, adj_59);
                wp::adj_add(var_51, var_58, adj_51, adj_56, adj_57);
                wp::adj_address(var_dof_A_diag_in, var_11, var_55, adj_dof_A_diag_in, adj_11, adj_55, adj_56);
                wp::adj_add(var_19, var_54, adj_19, adj_54, adj_55);
                // adj: + dof_A_diag_in[dof_A_diag_id, dofadr + 5]                                <L 3160>
                wp::adj_add(var_52, var_53, adj_47, adj_50, adj_51);
                wp::adj_address(var_dof_A_diag_in, var_11, var_49, adj_dof_A_diag_in, adj_11, adj_49, adj_50);
                wp::adj_add(var_19, var_48, adj_19, adj_48, adj_49);
                // adj: + dof_A_diag_in[dof_A_diag_id, dofadr + 4]                                <L 3159>
                wp::adj_address(var_dof_A_diag_in, var_11, var_46, adj_dof_A_diag_in, adj_11, adj_46, adj_47);
                wp::adj_add(var_19, var_45, adj_19, adj_45, adj_46);
                // adj: dof_A_diag_in[dof_A_diag_id, dofadr + 3]                                  <L 3158>
                // adj: avg = wp.static(1.0 / 3.0) * (                                            <L 3157>
            }
            if (var_27) {
                wp::adj_mul(var_28, var_41, adj_28, adj_41, adj_43);
                wp::adj_add(var_35, var_42, adj_35, adj_40, adj_41);
                wp::adj_address(var_dof_A_diag_in, var_11, var_39, adj_dof_A_diag_in, adj_11, adj_39, adj_40);
                wp::adj_add(var_19, var_38, adj_19, adj_38, adj_39);
                // adj: + dof_A_diag_in[dof_A_diag_id, dofadr + 2]                                <L 3154>
                wp::adj_add(var_36, var_37, adj_31, adj_34, adj_35);
                wp::adj_address(var_dof_A_diag_in, var_11, var_33, adj_dof_A_diag_in, adj_11, adj_33, adj_34);
                wp::adj_add(var_19, var_32, adj_19, adj_32, adj_33);
                // adj: + dof_A_diag_in[dof_A_diag_id, dofadr + 1]                                <L 3153>
                wp::adj_address(var_dof_A_diag_in, var_11, var_30, adj_dof_A_diag_in, adj_11, adj_30, adj_31);
                wp::adj_add(var_19, var_29, adj_19, adj_29, adj_30);
                // adj: dof_A_diag_in[dof_A_diag_id, dofadr + 0]                                  <L 3152>
                // adj: avg = wp.static(1.0 / 3.0) * (                                            <L 3151>
            }
            wp::adj_add(var_19, var_25, adj_19, adj_25, adj_26);
            // adj: if dofid < dofadr + 3:                                                        <L 3150>
        }
        // adj: if jtype == int(types.JointType.FREE.value):                                      <L 3148>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_jnt_dofadr, var_13, adj_jnt_dofadr, adj_13, adj_18);
        // adj: dofadr = jnt_dofadr[jntid]                                                        <L 3146>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_address(var_jnt_type, var_13, adj_jnt_type, adj_13, adj_15);
        // adj: jtype = jnt_type[jntid]                                                           <L 3145>
        wp::adj_copy(var_14, adj_12, adj_13);
        wp::adj_address(var_dof_jntid, var_1, adj_dof_jntid, adj_1, adj_12);
        // adj: jntid = dof_jntid[dofid]                                                          <L 3144>
        wp::adj_mod(var_0, var_9, adj_0, adj_9, adj_11);
        adj_dof_A_diag_in.shape = adj_7;
        // adj: dof_A_diag_id = worldid % dof_A_diag_in.shape[0]                                  <L 3142>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_dof_invweight0_out.shape = adj_2;
        // adj: dof_invweight0_id = worldid % dof_invweight0_out.shape[0]                         <L 3141>
        // adj: worldid, dofid = wp.tid()                                                         <L 3140>
        // adj: def _finalize_dof_invweight0(                                                     <L 3133>
        return;
}



extern "C" __global__ void _resolve_dampratio_9820ff1f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_dof_M0_in,
    wp::int32 var_nv,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm)
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
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::vec_t<10, wp::float32>* var_17;
        const wp::int32 var_18 = 0;
        wp::float32 var_19;
        wp::vec_t<10, wp::float32> var_20;
        wp::vec_t<10, wp::float32>* var_21;
        wp::vec_t<10, wp::float32> var_22;
        wp::vec_t<10, wp::float32> var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        const wp::float32 var_28 = 1e-15;
        bool var_29;
        const wp::int32 var_30 = 2;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::float32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::range_t var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        bool var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::float32 var_61 = 2.0;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::vec_t<10, wp::float32> var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 2;
        //---------
        // forward
        // def _resolve_dampratio(                                                                <L 3437>
        // worldid, actid = wp.tid()                                                              <L 3448>
        builtin_tid2d(var_0, var_1);
        // biastype = actuator_biastype[actid]                                                    <L 3449>
        var_2 = wp::address(var_actuator_biastype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if biastype != BiasType.AFFINE:                                                        <L 3452>
        var_6 = (var_3 != var_5);
        if (var_6) {
            // return                                                                             <L 3453>
            return;
        }
        // gainprm_id = worldid % actuator_gainprm.shape[0]                                       <L 3455>
        var_7 = &(var_actuator_gainprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // biasprm_id = worldid % actuator_biasprm.shape[0]                                       <L 3456>
        var_12 = &(var_actuator_biasprm.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // kp = actuator_gainprm[gainprm_id, actid][0]                                            <L 3457>
        var_17 = wp::address(var_actuator_gainprm, var_11, var_1);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        // biasprm = actuator_biasprm[biasprm_id, actid]                                          <L 3459>
        var_21 = wp::address(var_actuator_biasprm, var_16, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if wp.abs(kp + biasprm[1]) > MJ_MINVAL:                                                <L 3461>
        var_25 = wp::extract(var_22, var_24);
        var_26 = wp::add(var_19, var_25);
        var_27 = wp::abs(var_26);
        var_29 = (var_27 > var_28);
        if (var_29) {
            // return                                                                             <L 3462>
            return;
        }
        // if biasprm[2] <= 0.0:                                                                  <L 3463>
        var_31 = wp::extract(var_22, var_30);
        var_33 = (var_31 <= var_32);
        if (var_33) {
            // return                                                                             <L 3464>
            return;
        }
        // dampratio = biasprm[2]                                                                 <L 3466>
        var_35 = wp::extract(var_22, var_34);
        // mass = float(0.0)                                                                      <L 3469>
        var_37 = wp::float(var_36);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 3470>
        var_38 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 3471>
        var_41 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // for k in range(rownnz):                                                                <L 3472>
        var_44 = wp::range(var_39);
        start_for_3:;
            if (iter_cmp(var_44) == 0) goto end_for_3;
            var_45 = wp::iter_next(var_44);
            // sparseid = rowadr + k                                                              <L 3473>
            var_46 = wp::add(var_42, var_45);
            // j = moment_colind_in[worldid, sparseid]                                            <L 3474>
            var_47 = wp::address(var_moment_colind_in, var_0, var_46);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // moment = actuator_moment_in[worldid, sparseid]                                     <L 3475>
            var_50 = wp::address(var_actuator_moment_in, var_0, var_46);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // if wp.abs(moment) > MJ_MINVAL:                                                     <L 3476>
            var_53 = wp::abs(var_51);
            var_54 = (var_53 > var_28);
            if (var_54) {
                // mass += dof_M0_in[worldid, j] / (moment * moment)                              <L 3477>
                var_55 = wp::address(var_dof_M0_in, var_0, var_48);
                var_56 = wp::mul(var_51, var_51);
                var_58 = wp::load(var_55);
                var_57 = wp::div(var_58, var_56);
                var_59 = wp::add(var_37, var_57);
            }
            var_60 = wp::where(var_54, var_59, var_37);
            wp::assign(var_37, var_60);
            goto start_for_3;
        end_for_3:;
        // damping = dampratio * 2.0 * wp.sqrt(kp * mass)                                         <L 3479>
        var_62 = wp::mul(var_35, var_61);
        var_63 = wp::mul(var_19, var_37);
        var_64 = wp::sqrt(var_63);
        var_65 = wp::mul(var_62, var_64);
        // new_biasprm = biasprm                                                                  <L 3482>
        var_66 = wp::copy(var_22);
        // new_biasprm[2] = -damping                                                              <L 3483>
        var_67 = wp::neg(var_65);
        wp::assign_inplace(var_66, var_68, var_67);
        // actuator_biasprm[biasprm_id, actid] = new_biasprm                                      <L 3484>
        wp::array_store(var_actuator_biasprm, var_16, var_1, var_66);
}



extern "C" __global__ void _resolve_dampratio_9820ff1f_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_dof_M0_in,
    wp::int32 var_nv,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<wp::int32> adj_actuator_biastype,
    wp::array_t<wp::vec_t<10, wp::float32>> adj_actuator_gainprm,
    wp::array_t<wp::int32> adj_moment_rownnz_in,
    wp::array_t<wp::int32> adj_moment_rowadr_in,
    wp::array_t<wp::int32> adj_moment_colind_in,
    wp::array_t<wp::float32> adj_actuator_moment_in,
    wp::array_t<wp::float32> adj_dof_M0_in,
    wp::int32 adj_nv,
    wp::array_t<wp::vec_t<10, wp::float32>> adj_actuator_biasprm)
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
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::vec_t<10, wp::float32>* var_17;
        const wp::int32 var_18 = 0;
        wp::float32 var_19;
        wp::vec_t<10, wp::float32> var_20;
        wp::vec_t<10, wp::float32>* var_21;
        wp::vec_t<10, wp::float32> var_22;
        wp::vec_t<10, wp::float32> var_23;
        const wp::int32 var_24 = 1;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        const wp::float32 var_28 = 1e-15;
        bool var_29;
        const wp::int32 var_30 = 2;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::float32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::range_t var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        bool var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::float32 var_61 = 2.0;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::vec_t<10, wp::float32> var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 2;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::shape_t adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::shape_t adj_10 = {};
        wp::int32 adj_11 = {};
        wp::shape_t adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::shape_t adj_15 = {};
        wp::int32 adj_16 = {};
        wp::vec_t<10, wp::float32> adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::vec_t<10, wp::float32> adj_20 = {};
        wp::vec_t<10, wp::float32> adj_21 = {};
        wp::vec_t<10, wp::float32> adj_22 = {};
        wp::vec_t<10, wp::float32> adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        bool adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        bool adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::range_t adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        bool adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::vec_t<10, wp::float32> adj_66 = {};
        wp::float32 adj_67 = {};
        wp::int32 adj_68 = {};
        //---------
        // forward
        // def _resolve_dampratio(                                                                <L 3437>
        // worldid, actid = wp.tid()                                                              <L 3448>
        builtin_tid2d(var_0, var_1);
        // biastype = actuator_biastype[actid]                                                    <L 3449>
        var_2 = wp::address(var_actuator_biastype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if biastype != BiasType.AFFINE:                                                        <L 3452>
        var_6 = (var_3 != var_5);
        if (var_6) {
            // return                                                                             <L 3453>
            goto label0;
        }
        // gainprm_id = worldid % actuator_gainprm.shape[0]                                       <L 3455>
        var_7 = &(var_actuator_gainprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // biasprm_id = worldid % actuator_biasprm.shape[0]                                       <L 3456>
        var_12 = &(var_actuator_biasprm.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // kp = actuator_gainprm[gainprm_id, actid][0]                                            <L 3457>
        var_17 = wp::address(var_actuator_gainprm, var_11, var_1);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        // biasprm = actuator_biasprm[biasprm_id, actid]                                          <L 3459>
        var_21 = wp::address(var_actuator_biasprm, var_16, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if wp.abs(kp + biasprm[1]) > MJ_MINVAL:                                                <L 3461>
        var_25 = wp::extract(var_22, var_24);
        var_26 = wp::add(var_19, var_25);
        var_27 = wp::abs(var_26);
        var_29 = (var_27 > var_28);
        if (var_29) {
            // return                                                                             <L 3462>
            goto label1;
        }
        // if biasprm[2] <= 0.0:                                                                  <L 3463>
        var_31 = wp::extract(var_22, var_30);
        var_33 = (var_31 <= var_32);
        if (var_33) {
            // return                                                                             <L 3464>
            goto label2;
        }
        // dampratio = biasprm[2]                                                                 <L 3466>
        var_35 = wp::extract(var_22, var_34);
        // mass = float(0.0)                                                                      <L 3469>
        var_37 = wp::float(var_36);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 3470>
        var_38 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 3471>
        var_41 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // for k in range(rownnz):                                                                <L 3472>
        var_44 = wp::range(var_39);
        // damping = dampratio * 2.0 * wp.sqrt(kp * mass)                                         <L 3479>
        var_62 = wp::mul(var_35, var_61);
        var_63 = wp::mul(var_19, var_37);
        var_64 = wp::sqrt(var_63);
        var_65 = wp::mul(var_62, var_64);
        // new_biasprm = biasprm                                                                  <L 3482>
        var_66 = wp::copy(var_22);
        // new_biasprm[2] = -damping                                                              <L 3483>
        var_67 = wp::neg(var_65);
        wp::assign_inplace(var_66, var_68, var_67);
        // actuator_biasprm[biasprm_id, actid] = new_biasprm                                      <L 3484>
        // wp::array_store(var_actuator_biasprm, var_16, var_1, var_66);
        //---------
        // reverse
        wp::adj_array_store(var_actuator_biasprm, var_16, var_1, var_66, adj_actuator_biasprm, adj_16, adj_1, adj_66);
        // adj: actuator_biasprm[biasprm_id, actid] = new_biasprm                                 <L 3484>
        wp::adj_assign_inplace(var_66, var_68, var_67, adj_66, adj_68, adj_67);
        wp::adj_neg(var_65, adj_65, adj_67);
        // adj: new_biasprm[2] = -damping                                                         <L 3483>
        wp::adj_copy(var_22, adj_22, adj_66);
        // adj: new_biasprm = biasprm                                                             <L 3482>
        wp::adj_mul(var_62, var_64, adj_62, adj_64, adj_65);
        wp::adj_sqrt(var_63, var_64, adj_63, adj_64);
        wp::adj_mul(var_19, var_37, adj_19, adj_37, adj_63);
        wp::adj_mul(var_35, var_61, adj_35, adj_61, adj_62);
        // adj: damping = dampratio * 2.0 * wp.sqrt(kp * mass)                                    <L 3479>
        var_44 = wp::iter_reverse(var_44);
        start_for_3:;
            if (iter_cmp(var_44) == 0) goto end_for_3;
            var_45 = wp::iter_next(var_44);
        	adj_46 = {};
        	adj_47 = {};
        	adj_48 = {};
        	adj_49 = {};
        	adj_50 = {};
        	adj_51 = {};
        	adj_52 = {};
        	adj_53 = {};
        	adj_54 = {};
        	adj_55 = {};
        	adj_56 = {};
        	adj_57 = {};
        	adj_58 = {};
        	adj_59 = {};
        	adj_60 = {};
            // sparseid = rowadr + k                                                              <L 3473>
            var_46 = wp::add(var_42, var_45);
            // j = moment_colind_in[worldid, sparseid]                                            <L 3474>
            var_47 = wp::address(var_moment_colind_in, var_0, var_46);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // moment = actuator_moment_in[worldid, sparseid]                                     <L 3475>
            var_50 = wp::address(var_actuator_moment_in, var_0, var_46);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // if wp.abs(moment) > MJ_MINVAL:                                                     <L 3476>
            var_53 = wp::abs(var_51);
            var_54 = (var_53 > var_28);
            if (var_54) {
                // mass += dof_M0_in[worldid, j] / (moment * moment)                              <L 3477>
                var_55 = wp::address(var_dof_M0_in, var_0, var_48);
                var_56 = wp::mul(var_51, var_51);
                var_58 = wp::load(var_55);
                var_57 = wp::div(var_58, var_56);
                var_59 = wp::add(var_37, var_57);
            }
            var_60 = wp::where(var_54, var_59, var_37);
            wp::assign(var_37, var_60);
            wp::adj_assign(var_37, var_60, adj_37, adj_60);
            wp::adj_where(var_54, var_59, var_37, adj_54, adj_59, adj_37, adj_60);
            if (var_54) {
                wp::adj_add(var_37, var_57, adj_37, adj_57, adj_59);
                wp::adj_div(var_58, var_56, var_57, adj_55, adj_56, adj_57);
                wp::adj_mul(var_51, var_51, adj_51, adj_51, adj_56);
                wp::adj_address(var_dof_M0_in, var_0, var_48, adj_dof_M0_in, adj_0, adj_48, adj_55);
                // adj: mass += dof_M0_in[worldid, j] / (moment * moment)                         <L 3477>
            }
            wp::adj_abs(var_51, adj_51, adj_53);
            // adj: if wp.abs(moment) > MJ_MINVAL:                                                <L 3476>
            wp::adj_copy(var_52, adj_50, adj_51);
            wp::adj_address(var_actuator_moment_in, var_0, var_46, adj_actuator_moment_in, adj_0, adj_46, adj_50);
            // adj: moment = actuator_moment_in[worldid, sparseid]                                <L 3475>
            wp::adj_copy(var_49, adj_47, adj_48);
            wp::adj_address(var_moment_colind_in, var_0, var_46, adj_moment_colind_in, adj_0, adj_46, adj_47);
            // adj: j = moment_colind_in[worldid, sparseid]                                       <L 3474>
            wp::adj_add(var_42, var_45, adj_42, adj_45, adj_46);
            // adj: sparseid = rowadr + k                                                         <L 3473>
        	goto start_for_3;
        end_for_3:;
        // adj: for k in range(rownnz):                                                           <L 3472>
        wp::adj_copy(var_43, adj_41, adj_42);
        wp::adj_address(var_moment_rowadr_in, var_0, var_1, adj_moment_rowadr_in, adj_0, adj_1, adj_41);
        // adj: rowadr = moment_rowadr_in[worldid, actid]                                         <L 3471>
        wp::adj_copy(var_40, adj_38, adj_39);
        wp::adj_address(var_moment_rownnz_in, var_0, var_1, adj_moment_rownnz_in, adj_0, adj_1, adj_38);
        // adj: rownnz = moment_rownnz_in[worldid, actid]                                         <L 3470>
        wp::adj_float(var_36, adj_36, adj_37);
        // adj: mass = float(0.0)                                                                 <L 3469>
        wp::adj_extract(var_22, var_34, adj_22, adj_34, adj_35);
        // adj: dampratio = biasprm[2]                                                            <L 3466>
        if (var_33) {
            label2:;
            // adj: return                                                                        <L 3464>
        }
        wp::adj_extract(var_22, var_30, adj_22, adj_30, adj_31);
        // adj: if biasprm[2] <= 0.0:                                                             <L 3463>
        if (var_29) {
            label1:;
            // adj: return                                                                        <L 3462>
        }
        wp::adj_abs(var_26, adj_26, adj_27);
        wp::adj_add(var_19, var_25, adj_19, adj_25, adj_26);
        wp::adj_extract(var_22, var_24, adj_22, adj_24, adj_25);
        // adj: if wp.abs(kp + biasprm[1]) > MJ_MINVAL:                                           <L 3461>
        wp::adj_copy(var_23, adj_21, adj_22);
        wp::adj_address(var_actuator_biasprm, var_16, var_1, adj_actuator_biasprm, adj_16, adj_1, adj_21);
        // adj: biasprm = actuator_biasprm[biasprm_id, actid]                                     <L 3459>
        wp::adj_extract(var_20, var_18, adj_17, adj_18, adj_19);
        wp::adj_address(var_actuator_gainprm, var_11, var_1, adj_actuator_gainprm, adj_11, adj_1, adj_17);
        // adj: kp = actuator_gainprm[gainprm_id, actid][0]                                       <L 3457>
        wp::adj_mod(var_0, var_14, adj_0, adj_14, adj_16);
        adj_actuator_biasprm.shape = adj_12;
        // adj: biasprm_id = worldid % actuator_biasprm.shape[0]                                  <L 3456>
        wp::adj_mod(var_0, var_9, adj_0, adj_9, adj_11);
        adj_actuator_gainprm.shape = adj_7;
        // adj: gainprm_id = worldid % actuator_gainprm.shape[0]                                  <L 3455>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 3453>
        }
        // adj: if biastype != BiasType.AFFINE:                                                   <L 3452>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_actuator_biastype, var_1, adj_actuator_biastype, adj_1, adj_2);
        // adj: biastype = actuator_biastype[actid]                                               <L 3449>
        // adj: worldid, actid = wp.tid()                                                         <L 3448>
        // adj: def _resolve_dampratio(                                                           <L 3437>
        return;
}



extern "C" __global__ void _compute_actuator_acc0_a0c3fa27_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_actuator_acc0_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float32 var_2;
        wp::range_t var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _compute_actuator_acc0(                                                            <L 3408>
        // worldid = wp.tid()                                                                     <L 3414>
        var_0 = builtin_tid1d();
        // norm_sq = float(0.0)                                                                   <L 3415>
        var_2 = wp::float(var_1);
        // for i in range(nv):                                                                    <L 3416>
        var_3 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_3) == 0) goto end_for_0;
            var_4 = wp::iter_next(var_3);
            // norm_sq += result_vec_in[worldid, i] * result_vec_in[worldid, i]                   <L 3417>
            var_5 = wp::address(var_result_vec_in, var_0, var_4);
            var_6 = wp::address(var_result_vec_in, var_0, var_4);
            var_8 = wp::load(var_5);
            var_9 = wp::load(var_6);
            var_7 = wp::mul(var_8, var_9);
            var_10 = wp::add(var_2, var_7);
            wp::assign(var_2, var_10);
            goto start_for_0;
        end_for_0:;
        // actuator_acc0_out[worldid, actid_target] = wp.sqrt(norm_sq)                            <L 3418>
        var_11 = wp::sqrt(var_2);
        wp::array_store(var_actuator_acc0_out, var_0, var_actid_target, var_11);
}



extern "C" __global__ void _compute_actuator_acc0_a0c3fa27_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_actid_target,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_actuator_acc0_out,
    wp::int32 adj_actid_target,
    wp::int32 adj_nv,
    wp::array_t<wp::float32> adj_result_vec_in,
    wp::array_t<wp::float32> adj_actuator_acc0_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float32 var_2;
        wp::range_t var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::range_t adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        //---------
        // forward
        // def _compute_actuator_acc0(                                                            <L 3408>
        // worldid = wp.tid()                                                                     <L 3414>
        var_0 = builtin_tid1d();
        // norm_sq = float(0.0)                                                                   <L 3415>
        var_2 = wp::float(var_1);
        // for i in range(nv):                                                                    <L 3416>
        var_3 = wp::range(var_nv);
        // actuator_acc0_out[worldid, actid_target] = wp.sqrt(norm_sq)                            <L 3418>
        var_11 = wp::sqrt(var_2);
        // wp::array_store(var_actuator_acc0_out, var_0, var_actid_target, var_11);
        //---------
        // reverse
        wp::adj_array_store(var_actuator_acc0_out, var_0, var_actid_target, var_11, adj_actuator_acc0_out, adj_0, adj_actid_target, adj_11);
        wp::adj_sqrt(var_2, var_11, adj_2, adj_11);
        // adj: actuator_acc0_out[worldid, actid_target] = wp.sqrt(norm_sq)                       <L 3418>
        var_3 = wp::iter_reverse(var_3);
        start_for_0:;
            if (iter_cmp(var_3) == 0) goto end_for_0;
            var_4 = wp::iter_next(var_3);
        	adj_5 = {};
        	adj_6 = {};
        	adj_7 = {};
        	adj_8 = {};
        	adj_9 = {};
        	adj_10 = {};
            // norm_sq += result_vec_in[worldid, i] * result_vec_in[worldid, i]                   <L 3417>
            var_5 = wp::address(var_result_vec_in, var_0, var_4);
            var_6 = wp::address(var_result_vec_in, var_0, var_4);
            var_8 = wp::load(var_5);
            var_9 = wp::load(var_6);
            var_7 = wp::mul(var_8, var_9);
            var_10 = wp::add(var_2, var_7);
            wp::assign(var_2, var_10);
            wp::adj_assign(var_2, var_10, adj_2, adj_10);
            wp::adj_add(var_2, var_7, adj_2, adj_7, adj_10);
            wp::adj_mul(var_8, var_9, adj_5, adj_6, adj_7);
            wp::adj_address(var_result_vec_in, var_0, var_4, adj_result_vec_in, adj_0, adj_4, adj_6);
            wp::adj_address(var_result_vec_in, var_0, var_4, adj_result_vec_in, adj_0, adj_4, adj_5);
            // adj: norm_sq += result_vec_in[worldid, i] * result_vec_in[worldid, i]              <L 3417>
        	goto start_for_0;
        end_for_0:;
        // adj: for i in range(nv):                                                               <L 3416>
        wp::adj_float(var_1, adj_1, adj_2);
        // adj: norm_sq = float(0.0)                                                              <L 3415>
        // adj: worldid = wp.tid()                                                                <L 3414>
        // adj: def _compute_actuator_acc0(                                                       <L 3408>
        return;
}



extern "C" __global__ void _set_length_range_a26be645_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::int32> var_jnt_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_range,
    wp::array_t<wp::int32> var_tendon_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_range,
    wp::int32 var_ntendon,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange_out)
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
        wp::vec_t<2, wp::int32>* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::vec_t<2, wp::int32> var_8;
        wp::shape_t* var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::shape_t var_12;
        wp::int32 var_13;
        wp::vec_t<6, wp::float32>* var_14;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<2, wp::float32> var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        const wp::int32 var_24 = 1;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::shape_t* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::shape_t var_31;
        wp::int32 var_32;
        wp::vec_t<2, wp::float32>* var_33;
        wp::vec_t<2, wp::float32> var_34;
        wp::vec_t<2, wp::float32> var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<2, wp::float32> var_44;
        const wp::int32 var_45 = 1;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::vec_t<2, wp::float32> var_51;
        wp::vec_t<2, wp::float32> var_52;
        wp::int32 var_53;
        wp::vec_t<2, wp::float32> var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 3;
        bool var_57;
        bool var_58;
        const wp::int32 var_59 = 0;
        bool var_60;
        wp::int32* var_61;
        wp::int32 var_62;
        wp::shape_t* var_63;
        const wp::int32 var_64 = 0;
        wp::int32 var_65;
        wp::shape_t var_66;
        wp::int32 var_67;
        wp::vec_t<2, wp::float32>* var_68;
        wp::vec_t<2, wp::float32> var_69;
        wp::vec_t<2, wp::float32> var_70;
        const wp::float32 var_71 = 0.0;
        bool var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::int32 var_76 = 1;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<2, wp::float32> var_79;
        const wp::int32 var_80 = 1;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::vec_t<2, wp::float32> var_86;
        wp::vec_t<2, wp::float32> var_87;
        wp::vec_t<2, wp::float32> var_88;
        wp::vec_t<2, wp::float32> var_89;
        wp::vec_t<2, wp::float32> var_90;
        wp::vec_t<2, wp::float32> var_91;
        //---------
        // forward
        // def _set_length_range(                                                                 <L 3488>
        // worldid, actid = wp.tid()                                                              <L 3499>
        builtin_tid2d(var_0, var_1);
        // trntype = actuator_trntype[actid]                                                      <L 3500>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // id0 = actuator_trnid[actid][0]                                                         <L 3501>
        var_5 = wp::address(var_actuator_trnid, var_1);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        // gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]                      <L 3502>
        var_9 = &(var_actuator_gear.shape);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = wp::mod(var_0, var_11);
        var_14 = wp::address(var_actuator_gear, var_13, var_1);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        // lr = wp.vec2(0.0, 0.0)                                                                 <L 3504>
        var_20 = wp::vec_t<2, wp::float32>(var_18, var_19);
        // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 3506>
        var_23 = (var_3 == var_22);
        var_21 = var_23;
        if (!var_21) {
            var_25 = (var_3 == var_24);
            var_21 = var_21 || var_25;
        }
        if (var_21) {
            // if jnt_limited[id0]:                                                               <L 3507>
            var_26 = wp::address(var_jnt_limited, var_7);
            var_27 = wp::load(var_26);
            if (var_27) {
                // rng = jnt_range[worldid % jnt_range.shape[0], id0]                             <L 3508>
                var_28 = &(var_jnt_range.shape);
                var_31 = wp::load(var_28);
                var_30 = wp::extract(var_31, var_29);
                var_32 = wp::mod(var_0, var_30);
                var_33 = wp::address(var_jnt_range, var_32, var_7);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // if gear0 > 0.0:                                                                <L 3509>
                var_37 = (var_16 > var_36);
                if (var_37) {
                    // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                               <L 3510>
                    var_39 = wp::extract(var_34, var_38);
                    var_40 = wp::mul(var_39, var_16);
                    var_42 = wp::extract(var_34, var_41);
                    var_43 = wp::mul(var_42, var_16);
                    var_44 = wp::vec_t<2, wp::float32>(var_40, var_43);
                }
                if (!var_37) {
                    // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                               <L 3512>
                    var_46 = wp::extract(var_34, var_45);
                    var_47 = wp::mul(var_46, var_16);
                    var_49 = wp::extract(var_34, var_48);
                    var_50 = wp::mul(var_49, var_16);
                    var_51 = wp::vec_t<2, wp::float32>(var_47, var_50);
                }
                var_52 = wp::where(var_37, var_44, var_51);
            }
            var_53 = wp::load(var_26);
            var_55 = wp::load(var_26);
            var_54 = wp::where(var_55, var_52, var_20);
        }
        if (!var_21) {
            // elif trntype == TrnType.TENDON:                                                    <L 3513>
            var_57 = (var_3 == var_56);
            if (var_57) {
                // if ntendon > 0 and tendon_limited[id0]:                                        <L 3514>
                var_60 = (var_ntendon > var_59);
                var_58 = var_60;
                if (var_58) {
                    var_61 = wp::address(var_tendon_limited, var_7);
                    var_62 = wp::load(var_61);
                    var_58 = var_58 && var_62;
                }
                if (var_58) {
                    // rng = tendon_range[worldid % tendon_range.shape[0], id0]                   <L 3515>
                    var_63 = &(var_tendon_range.shape);
                    var_66 = wp::load(var_63);
                    var_65 = wp::extract(var_66, var_64);
                    var_67 = wp::mod(var_0, var_65);
                    var_68 = wp::address(var_tendon_range, var_67, var_7);
                    var_70 = wp::load(var_68);
                    var_69 = wp::copy(var_70);
                    // if gear0 > 0.0:                                                            <L 3516>
                    var_72 = (var_16 > var_71);
                    if (var_72) {
                        // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                           <L 3517>
                        var_74 = wp::extract(var_69, var_73);
                        var_75 = wp::mul(var_74, var_16);
                        var_77 = wp::extract(var_69, var_76);
                        var_78 = wp::mul(var_77, var_16);
                        var_79 = wp::vec_t<2, wp::float32>(var_75, var_78);
                    }
                    if (!var_72) {
                        // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                           <L 3519>
                        var_81 = wp::extract(var_69, var_80);
                        var_82 = wp::mul(var_81, var_16);
                        var_84 = wp::extract(var_69, var_83);
                        var_85 = wp::mul(var_84, var_16);
                        var_86 = wp::vec_t<2, wp::float32>(var_82, var_85);
                    }
                    var_87 = wp::where(var_72, var_79, var_86);
                }
                var_88 = wp::where(var_58, var_87, var_20);
            }
            var_89 = wp::where(var_57, var_88, var_20);
        }
        var_90 = wp::where(var_21, var_54, var_89);
        var_91 = wp::where(var_21, var_34, var_69);
        // actuator_lengthrange_out[worldid, actid] = lr                                          <L 3521>
        wp::array_store(var_actuator_lengthrange_out, var_0, var_1, var_90);
}



extern "C" __global__ void _set_length_range_a26be645_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::int32> var_jnt_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_range,
    wp::array_t<wp::int32> var_tendon_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_range,
    wp::int32 var_ntendon,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange_out,
    wp::array_t<wp::int32> adj_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> adj_actuator_trnid,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_actuator_gear,
    wp::array_t<wp::int32> adj_jnt_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_jnt_range,
    wp::array_t<wp::int32> adj_tendon_limited,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_tendon_range,
    wp::int32 adj_ntendon,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_actuator_lengthrange_out)
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
        wp::vec_t<2, wp::int32>* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::vec_t<2, wp::int32> var_8;
        wp::shape_t* var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::shape_t var_12;
        wp::int32 var_13;
        wp::vec_t<6, wp::float32>* var_14;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<2, wp::float32> var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        const wp::int32 var_24 = 1;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::shape_t* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::shape_t var_31;
        wp::int32 var_32;
        wp::vec_t<2, wp::float32>* var_33;
        wp::vec_t<2, wp::float32> var_34;
        wp::vec_t<2, wp::float32> var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<2, wp::float32> var_44;
        const wp::int32 var_45 = 1;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::vec_t<2, wp::float32> var_51;
        wp::vec_t<2, wp::float32> var_52;
        wp::int32 var_53;
        wp::vec_t<2, wp::float32> var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 3;
        bool var_57;
        bool var_58;
        const wp::int32 var_59 = 0;
        bool var_60;
        wp::int32* var_61;
        wp::int32 var_62;
        wp::shape_t* var_63;
        const wp::int32 var_64 = 0;
        wp::int32 var_65;
        wp::shape_t var_66;
        wp::int32 var_67;
        wp::vec_t<2, wp::float32>* var_68;
        wp::vec_t<2, wp::float32> var_69;
        wp::vec_t<2, wp::float32> var_70;
        const wp::float32 var_71 = 0.0;
        bool var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::int32 var_76 = 1;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<2, wp::float32> var_79;
        const wp::int32 var_80 = 1;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::vec_t<2, wp::float32> var_86;
        wp::vec_t<2, wp::float32> var_87;
        wp::vec_t<2, wp::float32> var_88;
        wp::vec_t<2, wp::float32> var_89;
        wp::vec_t<2, wp::float32> var_90;
        wp::vec_t<2, wp::float32> var_91;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::vec_t<2, wp::int32> adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<2, wp::int32> adj_8 = {};
        wp::shape_t adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::shape_t adj_12 = {};
        wp::int32 adj_13 = {};
        wp::vec_t<6, wp::float32> adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::vec_t<6, wp::float32> adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::vec_t<2, wp::float32> adj_20 = {};
        bool adj_21 = {};
        wp::int32 adj_22 = {};
        bool adj_23 = {};
        wp::int32 adj_24 = {};
        bool adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::shape_t adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::shape_t adj_31 = {};
        wp::int32 adj_32 = {};
        wp::vec_t<2, wp::float32> adj_33 = {};
        wp::vec_t<2, wp::float32> adj_34 = {};
        wp::vec_t<2, wp::float32> adj_35 = {};
        wp::float32 adj_36 = {};
        bool adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::vec_t<2, wp::float32> adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::float32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::vec_t<2, wp::float32> adj_51 = {};
        wp::vec_t<2, wp::float32> adj_52 = {};
        wp::int32 adj_53 = {};
        wp::vec_t<2, wp::float32> adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        bool adj_57 = {};
        bool adj_58 = {};
        wp::int32 adj_59 = {};
        bool adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::shape_t adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::shape_t adj_66 = {};
        wp::int32 adj_67 = {};
        wp::vec_t<2, wp::float32> adj_68 = {};
        wp::vec_t<2, wp::float32> adj_69 = {};
        wp::vec_t<2, wp::float32> adj_70 = {};
        wp::float32 adj_71 = {};
        bool adj_72 = {};
        wp::int32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::vec_t<2, wp::float32> adj_79 = {};
        wp::int32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::float32 adj_84 = {};
        wp::float32 adj_85 = {};
        wp::vec_t<2, wp::float32> adj_86 = {};
        wp::vec_t<2, wp::float32> adj_87 = {};
        wp::vec_t<2, wp::float32> adj_88 = {};
        wp::vec_t<2, wp::float32> adj_89 = {};
        wp::vec_t<2, wp::float32> adj_90 = {};
        wp::vec_t<2, wp::float32> adj_91 = {};
        //---------
        // forward
        // def _set_length_range(                                                                 <L 3488>
        // worldid, actid = wp.tid()                                                              <L 3499>
        builtin_tid2d(var_0, var_1);
        // trntype = actuator_trntype[actid]                                                      <L 3500>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // id0 = actuator_trnid[actid][0]                                                         <L 3501>
        var_5 = wp::address(var_actuator_trnid, var_1);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        // gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]                      <L 3502>
        var_9 = &(var_actuator_gear.shape);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = wp::mod(var_0, var_11);
        var_14 = wp::address(var_actuator_gear, var_13, var_1);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        // lr = wp.vec2(0.0, 0.0)                                                                 <L 3504>
        var_20 = wp::vec_t<2, wp::float32>(var_18, var_19);
        // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 3506>
        var_23 = (var_3 == var_22);
        var_21 = var_23;
        if (!var_21) {
            var_25 = (var_3 == var_24);
            var_21 = var_21 || var_25;
        }
        if (var_21) {
            // if jnt_limited[id0]:                                                               <L 3507>
            var_26 = wp::address(var_jnt_limited, var_7);
            var_27 = wp::load(var_26);
            if (var_27) {
                // rng = jnt_range[worldid % jnt_range.shape[0], id0]                             <L 3508>
                var_28 = &(var_jnt_range.shape);
                var_31 = wp::load(var_28);
                var_30 = wp::extract(var_31, var_29);
                var_32 = wp::mod(var_0, var_30);
                var_33 = wp::address(var_jnt_range, var_32, var_7);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // if gear0 > 0.0:                                                                <L 3509>
                var_37 = (var_16 > var_36);
                if (var_37) {
                    // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                               <L 3510>
                    var_39 = wp::extract(var_34, var_38);
                    var_40 = wp::mul(var_39, var_16);
                    var_42 = wp::extract(var_34, var_41);
                    var_43 = wp::mul(var_42, var_16);
                    var_44 = wp::vec_t<2, wp::float32>(var_40, var_43);
                }
                if (!var_37) {
                    // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                               <L 3512>
                    var_46 = wp::extract(var_34, var_45);
                    var_47 = wp::mul(var_46, var_16);
                    var_49 = wp::extract(var_34, var_48);
                    var_50 = wp::mul(var_49, var_16);
                    var_51 = wp::vec_t<2, wp::float32>(var_47, var_50);
                }
                var_52 = wp::where(var_37, var_44, var_51);
            }
            var_53 = wp::load(var_26);
            var_55 = wp::load(var_26);
            var_54 = wp::where(var_55, var_52, var_20);
        }
        if (!var_21) {
            // elif trntype == TrnType.TENDON:                                                    <L 3513>
            var_57 = (var_3 == var_56);
            if (var_57) {
                // if ntendon > 0 and tendon_limited[id0]:                                        <L 3514>
                var_60 = (var_ntendon > var_59);
                var_58 = var_60;
                if (var_58) {
                    var_61 = wp::address(var_tendon_limited, var_7);
                    var_62 = wp::load(var_61);
                    var_58 = var_58 && var_62;
                }
                if (var_58) {
                    // rng = tendon_range[worldid % tendon_range.shape[0], id0]                   <L 3515>
                    var_63 = &(var_tendon_range.shape);
                    var_66 = wp::load(var_63);
                    var_65 = wp::extract(var_66, var_64);
                    var_67 = wp::mod(var_0, var_65);
                    var_68 = wp::address(var_tendon_range, var_67, var_7);
                    var_70 = wp::load(var_68);
                    var_69 = wp::copy(var_70);
                    // if gear0 > 0.0:                                                            <L 3516>
                    var_72 = (var_16 > var_71);
                    if (var_72) {
                        // lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                           <L 3517>
                        var_74 = wp::extract(var_69, var_73);
                        var_75 = wp::mul(var_74, var_16);
                        var_77 = wp::extract(var_69, var_76);
                        var_78 = wp::mul(var_77, var_16);
                        var_79 = wp::vec_t<2, wp::float32>(var_75, var_78);
                    }
                    if (!var_72) {
                        // lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                           <L 3519>
                        var_81 = wp::extract(var_69, var_80);
                        var_82 = wp::mul(var_81, var_16);
                        var_84 = wp::extract(var_69, var_83);
                        var_85 = wp::mul(var_84, var_16);
                        var_86 = wp::vec_t<2, wp::float32>(var_82, var_85);
                    }
                    var_87 = wp::where(var_72, var_79, var_86);
                }
                var_88 = wp::where(var_58, var_87, var_20);
            }
            var_89 = wp::where(var_57, var_88, var_20);
        }
        var_90 = wp::where(var_21, var_54, var_89);
        var_91 = wp::where(var_21, var_34, var_69);
        // actuator_lengthrange_out[worldid, actid] = lr                                          <L 3521>
        // wp::array_store(var_actuator_lengthrange_out, var_0, var_1, var_90);
        //---------
        // reverse
        wp::adj_array_store(var_actuator_lengthrange_out, var_0, var_1, var_90, adj_actuator_lengthrange_out, adj_0, adj_1, adj_90);
        // adj: actuator_lengthrange_out[worldid, actid] = lr                                     <L 3521>
        wp::adj_where(var_21, var_34, var_69, adj_21, adj_34, adj_69, adj_91);
        wp::adj_where(var_21, var_54, var_89, adj_21, adj_54, adj_89, adj_90);
        if (!var_21) {
            wp::adj_where(var_57, var_88, var_20, adj_57, adj_88, adj_20, adj_89);
            if (var_57) {
                wp::adj_where(var_58, var_87, var_20, adj_58, adj_87, adj_20, adj_88);
                if (var_58) {
                    wp::adj_where(var_72, var_79, var_86, adj_72, adj_79, adj_86, adj_87);
                    if (!var_72) {
                        wp::adj_vec_t(var_82, var_85, adj_82, adj_85, adj_86);
                        wp::adj_mul(var_84, var_16, adj_84, adj_16, adj_85);
                        wp::adj_extract(var_69, var_83, adj_69, adj_83, adj_84);
                        wp::adj_mul(var_81, var_16, adj_81, adj_16, adj_82);
                        wp::adj_extract(var_69, var_80, adj_69, adj_80, adj_81);
                        // adj: lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                      <L 3519>
                    }
                    if (var_72) {
                        wp::adj_vec_t(var_75, var_78, adj_75, adj_78, adj_79);
                        wp::adj_mul(var_77, var_16, adj_77, adj_16, adj_78);
                        wp::adj_extract(var_69, var_76, adj_69, adj_76, adj_77);
                        wp::adj_mul(var_74, var_16, adj_74, adj_16, adj_75);
                        wp::adj_extract(var_69, var_73, adj_69, adj_73, adj_74);
                        // adj: lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                      <L 3517>
                    }
                    // adj: if gear0 > 0.0:                                                       <L 3516>
                    wp::adj_copy(var_70, adj_68, adj_69);
                    wp::adj_address(var_tendon_range, var_67, var_7, adj_tendon_range, adj_67, adj_7, adj_68);
                    wp::adj_mod(var_0, var_65, adj_0, adj_65, adj_67);
                    adj_tendon_range.shape = adj_63;
                    // adj: rng = tendon_range[worldid % tendon_range.shape[0], id0]              <L 3515>
                }
                if (var_58) {
                    wp::adj_address(var_tendon_limited, var_7, adj_tendon_limited, adj_7, adj_61);
                }
                // adj: if ntendon > 0 and tendon_limited[id0]:                                   <L 3514>
            }
            // adj: elif trntype == TrnType.TENDON:                                               <L 3513>
        }
        if (var_21) {
            wp::adj_where(var_55, var_52, var_20, adj_26, adj_52, adj_20, adj_54);
            if (var_53) {
                wp::adj_where(var_37, var_44, var_51, adj_37, adj_44, adj_51, adj_52);
                if (!var_37) {
                    wp::adj_vec_t(var_47, var_50, adj_47, adj_50, adj_51);
                    wp::adj_mul(var_49, var_16, adj_49, adj_16, adj_50);
                    wp::adj_extract(var_34, var_48, adj_34, adj_48, adj_49);
                    wp::adj_mul(var_46, var_16, adj_46, adj_16, adj_47);
                    wp::adj_extract(var_34, var_45, adj_34, adj_45, adj_46);
                    // adj: lr = wp.vec2(rng[1] * gear0, rng[0] * gear0)                          <L 3512>
                }
                if (var_37) {
                    wp::adj_vec_t(var_40, var_43, adj_40, adj_43, adj_44);
                    wp::adj_mul(var_42, var_16, adj_42, adj_16, adj_43);
                    wp::adj_extract(var_34, var_41, adj_34, adj_41, adj_42);
                    wp::adj_mul(var_39, var_16, adj_39, adj_16, adj_40);
                    wp::adj_extract(var_34, var_38, adj_34, adj_38, adj_39);
                    // adj: lr = wp.vec2(rng[0] * gear0, rng[1] * gear0)                          <L 3510>
                }
                // adj: if gear0 > 0.0:                                                           <L 3509>
                wp::adj_copy(var_35, adj_33, adj_34);
                wp::adj_address(var_jnt_range, var_32, var_7, adj_jnt_range, adj_32, adj_7, adj_33);
                wp::adj_mod(var_0, var_30, adj_0, adj_30, adj_32);
                adj_jnt_range.shape = adj_28;
                // adj: rng = jnt_range[worldid % jnt_range.shape[0], id0]                        <L 3508>
            }
            wp::adj_address(var_jnt_limited, var_7, adj_jnt_limited, adj_7, adj_26);
            // adj: if jnt_limited[id0]:                                                          <L 3507>
        }
        if (!var_21) {
        }
        // adj: if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                  <L 3506>
        wp::adj_vec_t(var_18, var_19, adj_18, adj_19, adj_20);
        // adj: lr = wp.vec2(0.0, 0.0)                                                            <L 3504>
        wp::adj_extract(var_17, var_15, adj_14, adj_15, adj_16);
        wp::adj_address(var_actuator_gear, var_13, var_1, adj_actuator_gear, adj_13, adj_1, adj_14);
        wp::adj_mod(var_0, var_11, adj_0, adj_11, adj_13);
        adj_actuator_gear.shape = adj_9;
        // adj: gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]                 <L 3502>
        wp::adj_extract(var_8, var_6, adj_5, adj_6, adj_7);
        wp::adj_address(var_actuator_trnid, var_1, adj_actuator_trnid, adj_1, adj_5);
        // adj: id0 = actuator_trnid[actid][0]                                                    <L 3501>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_actuator_trntype, var_1, adj_actuator_trntype, adj_1, adj_2);
        // adj: trntype = actuator_trntype[actid]                                                 <L 3500>
        // adj: worldid, actid = wp.tid()                                                         <L 3499>
        // adj: def _set_length_range(                                                            <L 3488>
        return;
}



extern "C" __global__ void _compute_light_pos0_a2f2ed64_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_light_bodyid,
    wp::array_t<wp::int32> var_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_poscom0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        //---------
        // forward
        // def _compute_light_pos0(                                                               <L 3361>
        // worldid, lightid = wp.tid()                                                            <L 3372>
        builtin_tid2d(var_0, var_1);
        // light_pos0_id = worldid % light_pos0_out.shape[0]                                      <L 3373>
        var_2 = &(var_light_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = light_bodyid[lightid]                                                         <L 3374>
        var_7 = wp::address(var_light_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = light_targetbodyid[lightid]                                                 <L 3375>
        var_10 = wp::address(var_light_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // light_xpos = light_xpos_in[worldid, lightid]                                           <L 3376>
        var_13 = wp::address(var_light_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // light_pos0_out[light_pos0_id, lightid] = light_xpos - xpos_in[worldid, bodyid]         <L 3378>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        wp::array_store(var_light_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 3379>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, targetid]       <L 3380>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            wp::array_store(var_light_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, bodyid]       <L 3382>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            wp::array_store(var_light_poscom0_out, var_6, var_1, var_25);
        }
        // light_dir0_out[light_pos0_id, lightid] = light_xdir_in[worldid, lightid]               <L 3383>
        var_27 = wp::address(var_light_xdir_in, var_0, var_1);
        var_28 = wp::load(var_27);
        wp::array_store(var_light_dir0_out, var_6, var_1, var_28);
}



extern "C" __global__ void _compute_light_pos0_a2f2ed64_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_light_bodyid,
    wp::array_t<wp::int32> var_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_poscom0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir0_out,
    wp::array_t<wp::int32> adj_light_bodyid,
    wp::array_t<wp::int32> adj_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_light_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_light_xdir_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_light_pos0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_light_poscom0_out,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_light_dir0_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::int32 adj_19 = {};
        bool adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        //---------
        // forward
        // def _compute_light_pos0(                                                               <L 3361>
        // worldid, lightid = wp.tid()                                                            <L 3372>
        builtin_tid2d(var_0, var_1);
        // light_pos0_id = worldid % light_pos0_out.shape[0]                                      <L 3373>
        var_2 = &(var_light_pos0_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = light_bodyid[lightid]                                                         <L 3374>
        var_7 = wp::address(var_light_bodyid, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // targetid = light_targetbodyid[lightid]                                                 <L 3375>
        var_10 = wp::address(var_light_targetbodyid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // light_xpos = light_xpos_in[worldid, lightid]                                           <L 3376>
        var_13 = wp::address(var_light_xpos_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // light_pos0_out[light_pos0_id, lightid] = light_xpos - xpos_in[worldid, bodyid]         <L 3378>
        var_16 = wp::address(var_xpos_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::sub(var_14, var_18);
        // wp::array_store(var_light_pos0_out, var_6, var_1, var_17);
        // if targetid >= 0:                                                                      <L 3379>
        var_20 = (var_11 >= var_19);
        if (var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, targetid]       <L 3380>
            var_21 = wp::address(var_subtree_com_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::sub(var_14, var_23);
            // wp::array_store(var_light_poscom0_out, var_6, var_1, var_22);
        }
        if (!var_20) {
            // light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, bodyid]       <L 3382>
            var_24 = wp::address(var_subtree_com_in, var_0, var_8);
            var_26 = wp::load(var_24);
            var_25 = wp::sub(var_14, var_26);
            // wp::array_store(var_light_poscom0_out, var_6, var_1, var_25);
        }
        // light_dir0_out[light_pos0_id, lightid] = light_xdir_in[worldid, lightid]               <L 3383>
        var_27 = wp::address(var_light_xdir_in, var_0, var_1);
        var_28 = wp::load(var_27);
        // wp::array_store(var_light_dir0_out, var_6, var_1, var_28);
        //---------
        // reverse
        wp::adj_array_store(var_light_dir0_out, var_6, var_1, var_28, adj_light_dir0_out, adj_6, adj_1, adj_27);
        wp::adj_address(var_light_xdir_in, var_0, var_1, adj_light_xdir_in, adj_0, adj_1, adj_27);
        // adj: light_dir0_out[light_pos0_id, lightid] = light_xdir_in[worldid, lightid]          <L 3383>
        if (!var_20) {
            wp::adj_array_store(var_light_poscom0_out, var_6, var_1, var_25, adj_light_poscom0_out, adj_6, adj_1, adj_25);
            wp::adj_sub(var_14, var_26, adj_14, adj_24, adj_25);
            wp::adj_address(var_subtree_com_in, var_0, var_8, adj_subtree_com_in, adj_0, adj_8, adj_24);
            // adj: light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, bodyid]  <L 3382>
        }
        if (var_20) {
            wp::adj_array_store(var_light_poscom0_out, var_6, var_1, var_22, adj_light_poscom0_out, adj_6, adj_1, adj_22);
            wp::adj_sub(var_14, var_23, adj_14, adj_21, adj_22);
            wp::adj_address(var_subtree_com_in, var_0, var_11, adj_subtree_com_in, adj_0, adj_11, adj_21);
            // adj: light_poscom0_out[light_pos0_id, lightid] = light_xpos - subtree_com_in[worldid, targetid]  <L 3380>
        }
        // adj: if targetid >= 0:                                                                 <L 3379>
        wp::adj_array_store(var_light_pos0_out, var_6, var_1, var_17, adj_light_pos0_out, adj_6, adj_1, adj_17);
        wp::adj_sub(var_14, var_18, adj_14, adj_16, adj_17);
        wp::adj_address(var_xpos_in, var_0, var_8, adj_xpos_in, adj_0, adj_8, adj_16);
        // adj: light_pos0_out[light_pos0_id, lightid] = light_xpos - xpos_in[worldid, bodyid]    <L 3378>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_light_xpos_in, var_0, var_1, adj_light_xpos_in, adj_0, adj_1, adj_13);
        // adj: light_xpos = light_xpos_in[worldid, lightid]                                      <L 3376>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_light_targetbodyid, var_1, adj_light_targetbodyid, adj_1, adj_10);
        // adj: targetid = light_targetbodyid[lightid]                                            <L 3375>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_light_bodyid, var_1, adj_light_bodyid, adj_1, adj_7);
        // adj: bodyid = light_bodyid[lightid]                                                    <L 3374>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_light_pos0_out.shape = adj_2;
        // adj: light_pos0_id = worldid % light_pos0_out.shape[0]                                 <L 3373>
        // adj: worldid, lightid = wp.tid()                                                       <L 3372>
        // adj: def _compute_light_pos0(                                                          <L 3361>
        return;
}



extern "C" __global__ void _accumulate_subtreemass_a3369753_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_subtreemass_io,
    wp::array_t<wp::int32> var_body_tree_)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def _accumulate_subtreemass(                                                           <L 2960>
        // worldid, nodeid = wp.tid()                                                             <L 2965>
        builtin_tid2d(var_0, var_1);
        // body_subtreemass_id = worldid % body_subtreemass_io.shape[0]                           <L 2966>
        var_2 = &(var_body_subtreemass_io.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = body_tree_[nodeid]                                                            <L 2967>
        var_7 = wp::address(var_body_tree_, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // parentid = body_parentid[bodyid]                                                       <L 2968>
        var_10 = wp::address(var_body_parentid, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if bodyid != 0:                                                                        <L 2969>
        var_14 = (var_8 != var_13);
        if (var_14) {
            // wp.atomic_add(body_subtreemass_io, body_subtreemass_id, parentid, body_subtreemass_io[body_subtreemass_id, bodyid])       <L 2970>
            var_15 = wp::address(var_body_subtreemass_io, var_6, var_8);
            var_17 = wp::load(var_15);
            var_16 = wp::atomic_add(var_body_subtreemass_io, var_6, var_11, var_17);
        }
}



extern "C" __global__ void _accumulate_subtreemass_a3369753_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_subtreemass_io,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::int32> adj_body_parentid,
    wp::array_t<wp::float32> adj_body_subtreemass_io,
    wp::array_t<wp::int32> adj_body_tree_)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        //---------
        // forward
        // def _accumulate_subtreemass(                                                           <L 2960>
        // worldid, nodeid = wp.tid()                                                             <L 2965>
        builtin_tid2d(var_0, var_1);
        // body_subtreemass_id = worldid % body_subtreemass_io.shape[0]                           <L 2966>
        var_2 = &(var_body_subtreemass_io.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // bodyid = body_tree_[nodeid]                                                            <L 2967>
        var_7 = wp::address(var_body_tree_, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // parentid = body_parentid[bodyid]                                                       <L 2968>
        var_10 = wp::address(var_body_parentid, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if bodyid != 0:                                                                        <L 2969>
        var_14 = (var_8 != var_13);
        if (var_14) {
            // wp.atomic_add(body_subtreemass_io, body_subtreemass_id, parentid, body_subtreemass_io[body_subtreemass_id, bodyid])       <L 2970>
            var_15 = wp::address(var_body_subtreemass_io, var_6, var_8);
            var_17 = wp::load(var_15);
            // var_16 = wp::atomic_add(var_body_subtreemass_io, var_6, var_11, var_17);
        }
        //---------
        // reverse
        if (var_14) {
            wp::adj_atomic_add(var_body_subtreemass_io, var_6, var_11, var_17, adj_body_subtreemass_io, adj_6, adj_11, adj_15, adj_16);
            wp::adj_address(var_body_subtreemass_io, var_6, var_8, adj_body_subtreemass_io, adj_6, adj_8, adj_15);
            // adj: wp.atomic_add(body_subtreemass_io, body_subtreemass_id, parentid, body_subtreemass_io[body_subtreemass_id, bodyid])  <L 2970>
        }
        // adj: if bodyid != 0:                                                                   <L 2969>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_body_parentid, var_8, adj_body_parentid, adj_8, adj_10);
        // adj: parentid = body_parentid[bodyid]                                                  <L 2968>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_body_tree_, var_1, adj_body_tree_, adj_1, adj_7);
        // adj: bodyid = body_tree_[nodeid]                                                       <L 2967>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_body_subtreemass_io.shape = adj_2;
        // adj: body_subtreemass_id = worldid % body_subtreemass_io.shape[0]                      <L 2966>
        // adj: worldid, nodeid = wp.tid()                                                        <L 2965>
        // adj: def _accumulate_subtreemass(                                                      <L 2960>
        return;
}



extern "C" __global__ void _init_subtreemass_a34720d1_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_mass_in,
    wp::array_t<wp::float32> var_body_subtreemass_out)
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
        wp::float32* var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def _init_subtreemass(                                                                 <L 2949>
        // worldid, bodyid = wp.tid()                                                             <L 2953>
        builtin_tid2d(var_0, var_1);
        // body_mass_id = worldid % body_mass_in.shape[0]                                         <L 2954>
        var_2 = &(var_body_mass_in.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_subtreemass_id = worldid % body_subtreemass_out.shape[0]                          <L 2955>
        var_7 = &(var_body_subtreemass_out.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // body_subtreemass_out[body_subtreemass_id, bodyid] = body_mass_in[body_mass_id, bodyid]       <L 2956>
        var_12 = wp::address(var_body_mass_in, var_6, var_1);
        var_13 = wp::load(var_12);
        wp::array_store(var_body_subtreemass_out, var_11, var_1, var_13);
}



extern "C" __global__ void _init_subtreemass_a34720d1_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_body_mass_in,
    wp::array_t<wp::float32> var_body_subtreemass_out,
    wp::array_t<wp::float32> adj_body_mass_in,
    wp::array_t<wp::float32> adj_body_subtreemass_out)
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
        wp::float32* var_12;
        wp::float32 var_13;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::shape_t adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::shape_t adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        //---------
        // forward
        // def _init_subtreemass(                                                                 <L 2949>
        // worldid, bodyid = wp.tid()                                                             <L 2953>
        builtin_tid2d(var_0, var_1);
        // body_mass_id = worldid % body_mass_in.shape[0]                                         <L 2954>
        var_2 = &(var_body_mass_in.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_subtreemass_id = worldid % body_subtreemass_out.shape[0]                          <L 2955>
        var_7 = &(var_body_subtreemass_out.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // body_subtreemass_out[body_subtreemass_id, bodyid] = body_mass_in[body_mass_id, bodyid]       <L 2956>
        var_12 = wp::address(var_body_mass_in, var_6, var_1);
        var_13 = wp::load(var_12);
        // wp::array_store(var_body_subtreemass_out, var_11, var_1, var_13);
        //---------
        // reverse
        wp::adj_array_store(var_body_subtreemass_out, var_11, var_1, var_13, adj_body_subtreemass_out, adj_11, adj_1, adj_12);
        wp::adj_address(var_body_mass_in, var_6, var_1, adj_body_mass_in, adj_6, adj_1, adj_12);
        // adj: body_subtreemass_out[body_subtreemass_id, bodyid] = body_mass_in[body_mass_id, bodyid]  <L 2956>
        wp::adj_mod(var_0, var_9, adj_0, adj_9, adj_11);
        adj_body_subtreemass_out.shape = adj_7;
        // adj: body_subtreemass_id = worldid % body_subtreemass_out.shape[0]                     <L 2955>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_body_mass_in.shape = adj_2;
        // adj: body_mass_id = worldid % body_mass_in.shape[0]                                    <L 2954>
        // adj: worldid, bodyid = wp.tid()                                                        <L 2953>
        // adj: def _init_subtreemass(                                                            <L 2949>
        return;
}



extern "C" __global__ void _resolve_tendon_lengthspring_a988a292_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::vec_t<2, wp::float32>* var_7;
        wp::vec_t<2, wp::float32> var_8;
        wp::vec_t<2, wp::float32> var_9;
        bool var_10;
        const wp::int32 var_11 = 0;
        wp::float32 var_12;
        const wp::float32 var_13 = -1.0;
        bool var_14;
        const wp::int32 var_15 = 1;
        wp::float32 var_16;
        const wp::float32 var_17 = -1.0;
        bool var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::vec_t<2, wp::float32> var_22;
        //---------
        // forward
        // def _resolve_tendon_lengthspring(                                                      <L 3071>
        // worldid, tenid = wp.tid()                                                              <L 3075>
        builtin_tid2d(var_0, var_1);
        // tendon_lengthspring_id = worldid % tendon_lengthspring_out.shape[0]                    <L 3076>
        var_2 = &(var_tendon_lengthspring_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // val = tendon_lengthspring_out[tendon_lengthspring_id, tenid]                           <L 3077>
        var_7 = wp::address(var_tendon_lengthspring_out, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if val[0] == -1.0 and val[1] == -1.0:                                                  <L 3078>
        var_12 = wp::extract(var_8, var_11);
        var_14 = (var_12 == var_13);
        var_10 = var_14;
        if (var_10) {
            var_16 = wp::extract(var_8, var_15);
            var_18 = (var_16 == var_17);
            var_10 = var_10 && var_18;
        }
        if (var_10) {
            // l = ten_length_in[worldid, tenid]                                                  <L 3079>
            var_19 = wp::address(var_ten_length_in, var_0, var_1);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // tendon_lengthspring_out[tendon_lengthspring_id, tenid] = wp.vec2(l, l)             <L 3080>
            var_22 = wp::vec_t<2, wp::float32>(var_20, var_20);
            wp::array_store(var_tendon_lengthspring_out, var_6, var_1, var_22);
        }
}



extern "C" __global__ void _resolve_tendon_lengthspring_a988a292_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring_out,
    wp::array_t<wp::float32> adj_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_tendon_lengthspring_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::vec_t<2, wp::float32>* var_7;
        wp::vec_t<2, wp::float32> var_8;
        wp::vec_t<2, wp::float32> var_9;
        bool var_10;
        const wp::int32 var_11 = 0;
        wp::float32 var_12;
        const wp::float32 var_13 = -1.0;
        bool var_14;
        const wp::int32 var_15 = 1;
        wp::float32 var_16;
        const wp::float32 var_17 = -1.0;
        bool var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::vec_t<2, wp::float32> var_22;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::vec_t<2, wp::float32> adj_7 = {};
        wp::vec_t<2, wp::float32> adj_8 = {};
        wp::vec_t<2, wp::float32> adj_9 = {};
        bool adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        bool adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::vec_t<2, wp::float32> adj_22 = {};
        //---------
        // forward
        // def _resolve_tendon_lengthspring(                                                      <L 3071>
        // worldid, tenid = wp.tid()                                                              <L 3075>
        builtin_tid2d(var_0, var_1);
        // tendon_lengthspring_id = worldid % tendon_lengthspring_out.shape[0]                    <L 3076>
        var_2 = &(var_tendon_lengthspring_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // val = tendon_lengthspring_out[tendon_lengthspring_id, tenid]                           <L 3077>
        var_7 = wp::address(var_tendon_lengthspring_out, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if val[0] == -1.0 and val[1] == -1.0:                                                  <L 3078>
        var_12 = wp::extract(var_8, var_11);
        var_14 = (var_12 == var_13);
        var_10 = var_14;
        if (var_10) {
            var_16 = wp::extract(var_8, var_15);
            var_18 = (var_16 == var_17);
            var_10 = var_10 && var_18;
        }
        if (var_10) {
            // l = ten_length_in[worldid, tenid]                                                  <L 3079>
            var_19 = wp::address(var_ten_length_in, var_0, var_1);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // tendon_lengthspring_out[tendon_lengthspring_id, tenid] = wp.vec2(l, l)             <L 3080>
            var_22 = wp::vec_t<2, wp::float32>(var_20, var_20);
            // wp::array_store(var_tendon_lengthspring_out, var_6, var_1, var_22);
        }
        //---------
        // reverse
        if (var_10) {
            wp::adj_array_store(var_tendon_lengthspring_out, var_6, var_1, var_22, adj_tendon_lengthspring_out, adj_6, adj_1, adj_22);
            wp::adj_vec_t(var_20, var_20, adj_20, adj_20, adj_22);
            // adj: tendon_lengthspring_out[tendon_lengthspring_id, tenid] = wp.vec2(l, l)        <L 3080>
            wp::adj_copy(var_21, adj_19, adj_20);
            wp::adj_address(var_ten_length_in, var_0, var_1, adj_ten_length_in, adj_0, adj_1, adj_19);
            // adj: l = ten_length_in[worldid, tenid]                                             <L 3079>
        }
        if (var_10) {
            wp::adj_extract(var_8, var_15, adj_8, adj_15, adj_16);
        }
        wp::adj_extract(var_8, var_11, adj_8, adj_11, adj_12);
        // adj: if val[0] == -1.0 and val[1] == -1.0:                                             <L 3078>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tendon_lengthspring_out, var_6, var_1, adj_tendon_lengthspring_out, adj_6, adj_1, adj_7);
        // adj: val = tendon_lengthspring_out[tendon_lengthspring_id, tenid]                      <L 3077>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_tendon_lengthspring_out.shape = adj_2;
        // adj: tendon_lengthspring_id = worldid % tendon_lengthspring_out.shape[0]               <L 3076>
        // adj: worldid, tenid = wp.tid()                                                         <L 3075>
        // adj: def _resolve_tendon_lengthspring(                                                 <L 3071>
        return;
}



extern "C" __global__ void _compute_meaninertia_af08211b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_M_rownnz_in,
    wp::array_t<wp::int32> var_M_rowadr_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_meaninertia_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        bool var_2;
        const wp::float32 var_3 = 1.0;
        wp::shape_t* var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::shape_t var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        //---------
        // forward
        // def _compute_meaninertia(                                                              <L 3084>
        // worldid = wp.tid()                                                                     <L 3092>
        var_0 = builtin_tid1d();
        // if nv == 0:                                                                            <L 3094>
        var_2 = (var_nv == var_1);
        if (var_2) {
            // meaninertia_out[worldid % meaninertia_out.shape[0]] = 1.0  # Default from MuJoCo       <L 3095>
            var_4 = &(var_meaninertia_out.shape);
            var_7 = wp::load(var_4);
            var_6 = wp::extract(var_7, var_5);
            var_8 = wp::mod(var_0, var_6);
            wp::array_store(var_meaninertia_out, var_8, var_3);
            // return                                                                             <L 3096>
            return;
        }
        // total = float(0.0)                                                                     <L 3098>
        var_10 = wp::float(var_9);
        // for i in range(nv):                                                                    <L 3099>
        var_11 = wp::range(var_nv);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
            // madr = M_rowadr_in[i] + M_rownnz_in[i] - 1                                         <L 3101>
            var_13 = wp::address(var_M_rowadr_in, var_12);
            var_14 = wp::address(var_M_rownnz_in, var_12);
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = wp::add(var_16, var_17);
            var_19 = wp::sub(var_15, var_18);
            // total += M_in[worldid, madr]                                                       <L 3102>
            var_20 = wp::address(var_M_in, var_0, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::add(var_10, var_22);
            wp::assign(var_10, var_21);
            goto start_for_1;
        end_for_1:;
        // meaninertia_out[worldid % meaninertia_out.shape[0]] = total / float(nv)                <L 3104>
        var_23 = wp::float(var_nv);
        var_24 = wp::div(var_10, var_23);
        var_25 = &(var_meaninertia_out.shape);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = wp::mod(var_0, var_27);
        wp::array_store(var_meaninertia_out, var_29, var_24);
}



extern "C" __global__ void _compute_meaninertia_af08211b_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_M_rownnz_in,
    wp::array_t<wp::int32> var_M_rowadr_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_meaninertia_out,
    wp::int32 adj_nv,
    wp::array_t<wp::int32> adj_M_rownnz_in,
    wp::array_t<wp::int32> adj_M_rowadr_in,
    wp::array_t<wp::float32> adj_M_in,
    wp::array_t<wp::float32> adj_meaninertia_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        bool var_2;
        const wp::float32 var_3 = 1.0;
        wp::shape_t* var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        wp::shape_t var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        wp::float32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::shape_t adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::range_t adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::shape_t adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::shape_t adj_28 = {};
        wp::int32 adj_29 = {};
        //---------
        // forward
        // def _compute_meaninertia(                                                              <L 3084>
        // worldid = wp.tid()                                                                     <L 3092>
        var_0 = builtin_tid1d();
        // if nv == 0:                                                                            <L 3094>
        var_2 = (var_nv == var_1);
        if (var_2) {
            // meaninertia_out[worldid % meaninertia_out.shape[0]] = 1.0  # Default from MuJoCo       <L 3095>
            var_4 = &(var_meaninertia_out.shape);
            var_7 = wp::load(var_4);
            var_6 = wp::extract(var_7, var_5);
            var_8 = wp::mod(var_0, var_6);
            // wp::array_store(var_meaninertia_out, var_8, var_3);
            // return                                                                             <L 3096>
            goto label0;
        }
        // total = float(0.0)                                                                     <L 3098>
        var_10 = wp::float(var_9);
        // for i in range(nv):                                                                    <L 3099>
        var_11 = wp::range(var_nv);
        // meaninertia_out[worldid % meaninertia_out.shape[0]] = total / float(nv)                <L 3104>
        var_23 = wp::float(var_nv);
        var_24 = wp::div(var_10, var_23);
        var_25 = &(var_meaninertia_out.shape);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = wp::mod(var_0, var_27);
        // wp::array_store(var_meaninertia_out, var_29, var_24);
        //---------
        // reverse
        wp::adj_array_store(var_meaninertia_out, var_29, var_24, adj_meaninertia_out, adj_29, adj_24);
        wp::adj_mod(var_0, var_27, adj_0, adj_27, adj_29);
        adj_meaninertia_out.shape = adj_25;
        wp::adj_div(var_10, var_23, var_24, adj_10, adj_23, adj_24);
        wp::adj_float(var_nv, adj_nv, adj_23);
        // adj: meaninertia_out[worldid % meaninertia_out.shape[0]] = total / float(nv)           <L 3104>
        var_11 = wp::iter_reverse(var_11);
        start_for_1:;
            if (iter_cmp(var_11) == 0) goto end_for_1;
            var_12 = wp::iter_next(var_11);
        	adj_13 = {};
        	adj_14 = {};
        	adj_15 = {};
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
        	adj_22 = {};
            // madr = M_rowadr_in[i] + M_rownnz_in[i] - 1                                         <L 3101>
            var_13 = wp::address(var_M_rowadr_in, var_12);
            var_14 = wp::address(var_M_rownnz_in, var_12);
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = wp::add(var_16, var_17);
            var_19 = wp::sub(var_15, var_18);
            // total += M_in[worldid, madr]                                                       <L 3102>
            var_20 = wp::address(var_M_in, var_0, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::add(var_10, var_22);
            wp::assign(var_10, var_21);
            wp::adj_assign(var_10, var_21, adj_10, adj_21);
            wp::adj_add(var_10, var_22, adj_10, adj_20, adj_21);
            wp::adj_address(var_M_in, var_0, var_19, adj_M_in, adj_0, adj_19, adj_20);
            // adj: total += M_in[worldid, madr]                                                  <L 3102>
            wp::adj_sub(var_15, var_18, adj_15, adj_18, adj_19);
            wp::adj_add(var_16, var_17, adj_13, adj_14, adj_15);
            wp::adj_address(var_M_rownnz_in, var_12, adj_M_rownnz_in, adj_12, adj_14);
            wp::adj_address(var_M_rowadr_in, var_12, adj_M_rowadr_in, adj_12, adj_13);
            // adj: madr = M_rowadr_in[i] + M_rownnz_in[i] - 1                                    <L 3101>
        	goto start_for_1;
        end_for_1:;
        // adj: for i in range(nv):                                                               <L 3099>
        wp::adj_float(var_9, adj_9, adj_10);
        // adj: total = float(0.0)                                                                <L 3098>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 3096>
            wp::adj_array_store(var_meaninertia_out, var_8, var_3, adj_meaninertia_out, adj_8, adj_3);
            wp::adj_mod(var_0, var_6, adj_0, adj_6, adj_8);
            adj_meaninertia_out.shape = adj_4;
            // adj: meaninertia_out[worldid % meaninertia_out.shape[0]] = 1.0  # Default from MuJoCo  <L 3095>
        }
        // adj: if nv == 0:                                                                       <L 3094>
        // adj: worldid = wp.tid()                                                                <L 3092>
        // adj: def _compute_meaninertia(                                                         <L 3084>
        return;
}



extern "C" __global__ void _compute_tendon_dot_product_be0bc0e6_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::int32 var_tenid_target,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_tendon_invweight0_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // forward
        // def _compute_tendon_dot_product(                                                       <L 3308>
        // worldid = wp.tid()                                                                     <L 3320>
        var_0 = builtin_tid1d();
        // tendon_invweight0_id = worldid % tendon_invweight0_out.shape[0]                        <L 3321>
        var_1 = &(var_tendon_invweight0_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 3322>
        var_7 = wp::float(var_6);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 3324>
        var_8 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 3325>
        var_11 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 3326>
        var_14 = wp::range(var_9);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
            // sparseid = rowadr + i                                                              <L 3327>
            var_16 = wp::add(var_12, var_15);
            // colind = ten_J_colind[sparseid]                                                    <L 3328>
            var_17 = wp::address(var_ten_J_colind, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // dot_prod += ten_J_in[worldid, sparseid] * result_vec_in[worldid, colind]           <L 3329>
            var_20 = wp::address(var_ten_J_in, var_0, var_16);
            var_21 = wp::address(var_result_vec_in, var_0, var_18);
            var_23 = wp::load(var_20);
            var_24 = wp::load(var_21);
            var_22 = wp::mul(var_23, var_24);
            var_25 = wp::add(var_7, var_22);
            wp::assign(var_7, var_25);
            goto start_for_0;
        end_for_0:;
        // tendon_invweight0_out[tendon_invweight0_id, tenid_target] = dot_prod                   <L 3331>
        wp::array_store(var_tendon_invweight0_out, var_5, var_tenid_target, var_7);
}



extern "C" __global__ void _compute_tendon_dot_product_be0bc0e6_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::int32 var_tenid_target,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_tendon_invweight0_out,
    wp::array_t<wp::int32> adj_ten_J_rownnz,
    wp::array_t<wp::int32> adj_ten_J_rowadr,
    wp::array_t<wp::int32> adj_ten_J_colind,
    wp::int32 adj_tenid_target,
    wp::array_t<wp::float32> adj_ten_J_in,
    wp::array_t<wp::float32> adj_result_vec_in,
    wp::array_t<wp::float32> adj_tendon_invweight0_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::shape_t adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::range_t adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        //---------
        // forward
        // def _compute_tendon_dot_product(                                                       <L 3308>
        // worldid = wp.tid()                                                                     <L 3320>
        var_0 = builtin_tid1d();
        // tendon_invweight0_id = worldid % tendon_invweight0_out.shape[0]                        <L 3321>
        var_1 = &(var_tendon_invweight0_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 3322>
        var_7 = wp::float(var_6);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 3324>
        var_8 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 3325>
        var_11 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(rownnz):                                                                <L 3326>
        var_14 = wp::range(var_9);
        // tendon_invweight0_out[tendon_invweight0_id, tenid_target] = dot_prod                   <L 3331>
        // wp::array_store(var_tendon_invweight0_out, var_5, var_tenid_target, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_tendon_invweight0_out, var_5, var_tenid_target, var_7, adj_tendon_invweight0_out, adj_5, adj_tenid_target, adj_7);
        // adj: tendon_invweight0_out[tendon_invweight0_id, tenid_target] = dot_prod              <L 3331>
        var_14 = wp::iter_reverse(var_14);
        start_for_0:;
            if (iter_cmp(var_14) == 0) goto end_for_0;
            var_15 = wp::iter_next(var_14);
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
        	adj_22 = {};
        	adj_23 = {};
        	adj_24 = {};
        	adj_25 = {};
            // sparseid = rowadr + i                                                              <L 3327>
            var_16 = wp::add(var_12, var_15);
            // colind = ten_J_colind[sparseid]                                                    <L 3328>
            var_17 = wp::address(var_ten_J_colind, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // dot_prod += ten_J_in[worldid, sparseid] * result_vec_in[worldid, colind]           <L 3329>
            var_20 = wp::address(var_ten_J_in, var_0, var_16);
            var_21 = wp::address(var_result_vec_in, var_0, var_18);
            var_23 = wp::load(var_20);
            var_24 = wp::load(var_21);
            var_22 = wp::mul(var_23, var_24);
            var_25 = wp::add(var_7, var_22);
            wp::assign(var_7, var_25);
            wp::adj_assign(var_7, var_25, adj_7, adj_25);
            wp::adj_add(var_7, var_22, adj_7, adj_22, adj_25);
            wp::adj_mul(var_23, var_24, adj_20, adj_21, adj_22);
            wp::adj_address(var_result_vec_in, var_0, var_18, adj_result_vec_in, adj_0, adj_18, adj_21);
            wp::adj_address(var_ten_J_in, var_0, var_16, adj_ten_J_in, adj_0, adj_16, adj_20);
            // adj: dot_prod += ten_J_in[worldid, sparseid] * result_vec_in[worldid, colind]      <L 3329>
            wp::adj_copy(var_19, adj_17, adj_18);
            wp::adj_address(var_ten_J_colind, var_16, adj_ten_J_colind, adj_16, adj_17);
            // adj: colind = ten_J_colind[sparseid]                                               <L 3328>
            wp::adj_add(var_12, var_15, adj_12, adj_15, adj_16);
            // adj: sparseid = rowadr + i                                                         <L 3327>
        	goto start_for_0;
        end_for_0:;
        // adj: for i in range(rownnz):                                                           <L 3326>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_ten_J_rowadr, var_tenid_target, adj_ten_J_rowadr, adj_tenid_target, adj_11);
        // adj: rowadr = ten_J_rowadr[tenid_target]                                               <L 3325>
        wp::adj_copy(var_10, adj_8, adj_9);
        wp::adj_address(var_ten_J_rownnz, var_tenid_target, adj_ten_J_rownnz, adj_tenid_target, adj_8);
        // adj: rownnz = ten_J_rownnz[tenid_target]                                               <L 3324>
        wp::adj_float(var_6, adj_6, adj_7);
        // adj: dot_prod = float(0.0)                                                             <L 3322>
        wp::adj_mod(var_0, var_3, adj_0, adj_3, adj_5);
        adj_tendon_invweight0_out.shape = adj_1;
        // adj: tendon_invweight0_id = worldid % tendon_invweight0_out.shape[0]                   <L 3321>
        // adj: worldid = wp.tid()                                                                <L 3320>
        // adj: def _compute_tendon_dot_product(                                                  <L 3308>
        return;
}



extern "C" __global__ void _copy_tendon_jacobian_cf465fd5_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_tenid_target,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_J_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 2;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def _copy_tendon_jacobian(                                                             <L 3290>
        // worldid = wp.tid()                                                                     <L 3298>
        var_0 = builtin_tid1d();
        // nv = ten_J_in.shape[2]                                                                 <L 3299>
        var_1 = &(var_ten_J_in.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 3300>
        var_5 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 3301>
        var_8 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for i in range(rownnz):                                                                <L 3302>
        var_11 = wp::range(var_6);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
            // colind = ten_J_colind[rowadr + i]                                                  <L 3303>
            var_13 = wp::add(var_9, var_12);
            var_14 = wp::address(var_ten_J_colind, var_13);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // ten_J_vec_out[worldid, colind] = ten_J_in[worldid, rowadr + i]                     <L 3304>
            var_17 = wp::add(var_9, var_12);
            var_18 = wp::address(var_ten_J_in, var_0, var_17);
            var_19 = wp::load(var_18);
            wp::array_store(var_ten_J_vec_out, var_0, var_15, var_19);
            goto start_for_0;
        end_for_0:;
}



extern "C" __global__ void _copy_tendon_jacobian_cf465fd5_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_tenid_target,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_J_vec_out,
    wp::int32 adj_tenid_target,
    wp::array_t<wp::int32> adj_ten_J_rownnz,
    wp::array_t<wp::int32> adj_ten_J_rowadr,
    wp::array_t<wp::int32> adj_ten_J_colind,
    wp::array_t<wp::float32> adj_ten_J_in,
    wp::array_t<wp::float32> adj_ten_J_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 2;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::shape_t adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::range_t adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        //---------
        // forward
        // def _copy_tendon_jacobian(                                                             <L 3290>
        // worldid = wp.tid()                                                                     <L 3298>
        var_0 = builtin_tid1d();
        // nv = ten_J_in.shape[2]                                                                 <L 3299>
        var_1 = &(var_ten_J_in.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // rownnz = ten_J_rownnz[tenid_target]                                                    <L 3300>
        var_5 = wp::address(var_ten_J_rownnz, var_tenid_target);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // rowadr = ten_J_rowadr[tenid_target]                                                    <L 3301>
        var_8 = wp::address(var_ten_J_rowadr, var_tenid_target);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for i in range(rownnz):                                                                <L 3302>
        var_11 = wp::range(var_6);
        //---------
        // reverse
        var_11 = wp::iter_reverse(var_11);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
        	adj_13 = {};
        	adj_14 = {};
        	adj_15 = {};
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
        	adj_19 = {};
            // colind = ten_J_colind[rowadr + i]                                                  <L 3303>
            var_13 = wp::add(var_9, var_12);
            var_14 = wp::address(var_ten_J_colind, var_13);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // ten_J_vec_out[worldid, colind] = ten_J_in[worldid, rowadr + i]                     <L 3304>
            var_17 = wp::add(var_9, var_12);
            var_18 = wp::address(var_ten_J_in, var_0, var_17);
            var_19 = wp::load(var_18);
            // wp::array_store(var_ten_J_vec_out, var_0, var_15, var_19);
            wp::adj_array_store(var_ten_J_vec_out, var_0, var_15, var_19, adj_ten_J_vec_out, adj_0, adj_15, adj_18);
            wp::adj_address(var_ten_J_in, var_0, var_17, adj_ten_J_in, adj_0, adj_17, adj_18);
            wp::adj_add(var_9, var_12, adj_9, adj_12, adj_17);
            // adj: ten_J_vec_out[worldid, colind] = ten_J_in[worldid, rowadr + i]                <L 3304>
            wp::adj_copy(var_16, adj_14, adj_15);
            wp::adj_address(var_ten_J_colind, var_13, adj_ten_J_colind, adj_13, adj_14);
            wp::adj_add(var_9, var_12, adj_9, adj_12, adj_13);
            // adj: colind = ten_J_colind[rowadr + i]                                             <L 3303>
        	goto start_for_0;
        end_for_0:;
        // adj: for i in range(rownnz):                                                           <L 3302>
        wp::adj_copy(var_10, adj_8, adj_9);
        wp::adj_address(var_ten_J_rowadr, var_tenid_target, adj_ten_J_rowadr, adj_tenid_target, adj_8);
        // adj: rowadr = ten_J_rowadr[tenid_target]                                               <L 3301>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_ten_J_rownnz, var_tenid_target, adj_ten_J_rownnz, adj_tenid_target, adj_5);
        // adj: rownnz = ten_J_rownnz[tenid_target]                                               <L 3300>
        adj_ten_J_in.shape = adj_1;
        // adj: nv = ten_J_in.shape[2]                                                            <L 3299>
        // adj: worldid = wp.tid()                                                                <L 3298>
        // adj: def _copy_tendon_jacobian(                                                        <L 3290>
        return;
}



extern "C" __global__ void _compute_eq_data0_d31148f7_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<11, wp::float32>* var_13;
        wp::vec_t<11, wp::float32> var_14;
        wp::vec_t<11, wp::float32> var_15;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        bool var_19;
        const wp::int32 var_20 = 1;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::mat_t<3, 3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::mat_t<3, 3, wp::float32>* var_43;
        wp::mat_t<3, 3, wp::float32> var_44;
        wp::mat_t<3, 3, wp::float32> var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        const wp::int32 var_50 = 0;
        wp::float32 var_51;
        const wp::int32 var_52 = 3;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        const wp::int32 var_55 = 4;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 5;
        const wp::int32 var_59 = 6;
        const wp::int32 var_60 = 6;
        wp::int32 var_61;
        bool var_62;
        const wp::float32 var_63 = 0.0;
        wp::vec_t<11, wp::float32> var_64;
        const wp::int32 var_65 = 1;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        bool var_68;
        const wp::int32 var_69 = 1;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        bool var_72;
        const wp::int32 var_73 = 6;
        wp::float32 var_74;
        const wp::int32 var_75 = 7;
        wp::float32 var_76;
        const wp::int32 var_77 = 8;
        wp::float32 var_78;
        const wp::int32 var_79 = 9;
        wp::float32 var_80;
        wp::quat_t<wp::float32> var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        wp::quat_t<wp::float32> var_85;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        const wp::int32 var_88 = 6;
        const wp::int32 var_89 = 1;
        wp::float32 var_90;
        const wp::int32 var_91 = 7;
        const wp::int32 var_92 = 2;
        wp::float32 var_93;
        const wp::int32 var_94 = 8;
        const wp::int32 var_95 = 3;
        wp::float32 var_96;
        const wp::int32 var_97 = 9;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        const wp::int32 var_106 = 1;
        wp::float32 var_107;
        const wp::int32 var_108 = 2;
        wp::float32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32>* var_111;
        wp::mat_t<3, 3, wp::float32>* var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::mat_t<3, 3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::mat_t<3, 3, wp::float32>* var_117;
        wp::mat_t<3, 3, wp::float32> var_118;
        wp::mat_t<3, 3, wp::float32> var_119;
        wp::vec_t<3, wp::float32>* var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::vec_t<3, wp::float32> var_123;
        const wp::int32 var_124 = 0;
        wp::float32 var_125;
        const wp::int32 var_126 = 3;
        const wp::int32 var_127 = 1;
        wp::float32 var_128;
        const wp::int32 var_129 = 4;
        const wp::int32 var_130 = 2;
        wp::float32 var_131;
        const wp::int32 var_132 = 5;
        wp::quat_t<wp::float32>* var_133;
        wp::quat_t<wp::float32> var_134;
        wp::quat_t<wp::float32> var_135;
        wp::quat_t<wp::float32>* var_136;
        wp::quat_t<wp::float32> var_137;
        wp::quat_t<wp::float32> var_138;
        const wp::int32 var_139 = 0;
        wp::float32 var_140;
        const wp::int32 var_141 = 6;
        const wp::int32 var_142 = 1;
        wp::float32 var_143;
        const wp::int32 var_144 = 7;
        const wp::int32 var_145 = 2;
        wp::float32 var_146;
        const wp::int32 var_147 = 8;
        const wp::int32 var_148 = 3;
        wp::float32 var_149;
        const wp::int32 var_150 = 9;
        wp::quat_t<wp::float32> var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        //---------
        // forward
        // def _compute_eq_data0(                                                                 <L 2994>
        // worldid, eqid = wp.tid()                                                               <L 3011>
        builtin_tid2d(var_0, var_1);
        // eq_data_id = worldid % eq_data_out.shape[0]                                            <L 3012>
        var_2 = &(var_eq_data_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // eqtype = eq_type[eqid]                                                                 <L 3014>
        var_7 = wp::address(var_eq_type, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // objtype = eq_objtype[eqid]                                                             <L 3015>
        var_10 = wp::address(var_eq_objtype, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // data = eq_data_out[eq_data_id, eqid]                                                   <L 3016>
        var_13 = wp::address(var_eq_data_out, var_6, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if eqtype == int(types.EqType.CONNECT.value):                                          <L 3018>
        var_18 = wp::int(var_17);
        var_19 = (var_8 == var_18);
        if (var_19) {
            // if objtype == int(types.ObjType.BODY.value):                                       <L 3019>
            var_22 = wp::int(var_21);
            var_23 = (var_11 == var_22);
            if (var_23) {
                // obj1id = eq_obj1id[eqid]                                                       <L 3020>
                var_24 = wp::address(var_eq_obj1id, var_1);
                var_26 = wp::load(var_24);
                var_25 = wp::copy(var_26);
                // obj2id = eq_obj2id[eqid]                                                       <L 3021>
                var_27 = wp::address(var_eq_obj2id, var_1);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // anchor1 = wp.vec3(data[0], data[1], data[2])                                   <L 3024>
                var_31 = wp::extract(var_14, var_30);
                var_33 = wp::extract(var_14, var_32);
                var_35 = wp::extract(var_14, var_34);
                var_36 = wp::vec_t<3, wp::float32>(var_31, var_33, var_35);
                // pos = xpos_in[worldid, obj1id] + xmat_in[worldid, obj1id] @ anchor1            <L 3025>
                var_37 = wp::address(var_xpos_in, var_0, var_25);
                var_38 = wp::address(var_xmat_in, var_0, var_25);
                var_40 = wp::load(var_38);
                var_39 = wp::mul(var_40, var_36);
                var_42 = wp::load(var_37);
                var_41 = wp::add(var_42, var_39);
                // anchor2 = wp.transpose(xmat_in[worldid, obj2id]) @ (pos - xpos_in[worldid, obj2id])       <L 3028>
                var_43 = wp::address(var_xmat_in, var_0, var_28);
                var_45 = wp::load(var_43);
                var_44 = wp::transpose(var_45);
                var_46 = wp::address(var_xpos_in, var_0, var_28);
                var_48 = wp::load(var_46);
                var_47 = wp::sub(var_41, var_48);
                var_49 = wp::mul(var_44, var_47);
                // data[3] = anchor2[0]                                                           <L 3029>
                var_51 = wp::extract(var_49, var_50);
                wp::assign_inplace(var_14, var_52, var_51);
                // data[4] = anchor2[1]                                                           <L 3030>
                var_54 = wp::extract(var_49, var_53);
                wp::assign_inplace(var_14, var_55, var_54);
                // data[5] = anchor2[2]                                                           <L 3031>
                var_57 = wp::extract(var_49, var_56);
                wp::assign_inplace(var_14, var_58, var_57);
                // eq_data_out[eq_data_id, eqid] = data                                           <L 3032>
                wp::array_store(var_eq_data_out, var_6, var_1, var_14);
            }
            if (!var_23) {
                // elif objtype == int(types.ObjType.SITE.value):                                 <L 3033>
                var_61 = wp::int(var_60);
                var_62 = (var_11 == var_61);
                if (var_62) {
                    // eq_data_out[eq_data_id, eqid] = types.vec11(0.0)                           <L 3035>
                    var_64 = wp::vec_t<11, wp::float32>(var_63);
                    wp::array_store(var_eq_data_out, var_6, var_1, var_64);
                }
            }
        }
        if (!var_19) {
            // elif eqtype == int(types.EqType.WELD.value):                                       <L 3036>
            var_67 = wp::int(var_66);
            var_68 = (var_8 == var_67);
            if (var_68) {
                // if objtype == int(types.ObjType.BODY.value):                                   <L 3037>
                var_71 = wp::int(var_70);
                var_72 = (var_11 == var_71);
                if (var_72) {
                    // quat = wp.quat(data[6], data[7], data[8], data[9])                         <L 3038>
                    var_74 = wp::extract(var_14, var_73);
                    var_76 = wp::extract(var_14, var_75);
                    var_78 = wp::extract(var_14, var_77);
                    var_80 = wp::extract(var_14, var_79);
                    var_81 = wp::quat_t<wp::float32>(var_74, var_76, var_78, var_80);
                    // if wp.length_sq(quat) > 0.0:                                               <L 3039>
                    var_82 = wp::length_sq(var_81);
                    var_84 = (var_82 > var_83);
                    if (var_84) {
                        // quat = wp.normalize(quat)                                              <L 3041>
                        var_85 = wp::normalize(var_81);
                        // data[6] = quat[0]                                                      <L 3042>
                        var_87 = wp::extract(var_85, var_86);
                        wp::assign_inplace(var_14, var_88, var_87);
                        // data[7] = quat[1]                                                      <L 3043>
                        var_90 = wp::extract(var_85, var_89);
                        wp::assign_inplace(var_14, var_91, var_90);
                        // data[8] = quat[2]                                                      <L 3044>
                        var_93 = wp::extract(var_85, var_92);
                        wp::assign_inplace(var_14, var_94, var_93);
                        // data[9] = quat[3]                                                      <L 3045>
                        var_96 = wp::extract(var_85, var_95);
                        wp::assign_inplace(var_14, var_97, var_96);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 3046>
                        wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    if (!var_84) {
                        // obj1id = eq_obj1id[eqid]                                               <L 3048>
                        var_98 = wp::address(var_eq_obj1id, var_1);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // obj2id = eq_obj2id[eqid]                                               <L 3049>
                        var_101 = wp::address(var_eq_obj2id, var_1);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // anchor2 = wp.vec3(data[0], data[1], data[2])                           <L 3052>
                        var_105 = wp::extract(var_14, var_104);
                        var_107 = wp::extract(var_14, var_106);
                        var_109 = wp::extract(var_14, var_108);
                        var_110 = wp::vec_t<3, wp::float32>(var_105, var_107, var_109);
                        // pos = xpos_in[worldid, obj2id] + xmat_in[worldid, obj2id] @ anchor2       <L 3053>
                        var_111 = wp::address(var_xpos_in, var_0, var_102);
                        var_112 = wp::address(var_xmat_in, var_0, var_102);
                        var_114 = wp::load(var_112);
                        var_113 = wp::mul(var_114, var_110);
                        var_116 = wp::load(var_111);
                        var_115 = wp::add(var_116, var_113);
                        // anchor1 = wp.transpose(xmat_in[worldid, obj1id]) @ (pos - xpos_in[worldid, obj1id])       <L 3056>
                        var_117 = wp::address(var_xmat_in, var_0, var_99);
                        var_119 = wp::load(var_117);
                        var_118 = wp::transpose(var_119);
                        var_120 = wp::address(var_xpos_in, var_0, var_99);
                        var_122 = wp::load(var_120);
                        var_121 = wp::sub(var_115, var_122);
                        var_123 = wp::mul(var_118, var_121);
                        // data[3] = anchor1[0]                                                   <L 3057>
                        var_125 = wp::extract(var_123, var_124);
                        wp::assign_inplace(var_14, var_126, var_125);
                        // data[4] = anchor1[1]                                                   <L 3058>
                        var_128 = wp::extract(var_123, var_127);
                        wp::assign_inplace(var_14, var_129, var_128);
                        // data[5] = anchor1[2]                                                   <L 3059>
                        var_131 = wp::extract(var_123, var_130);
                        wp::assign_inplace(var_14, var_132, var_131);
                        // relquat = mjmath.mul_quat(mjmath.quat_inv(xquat_in[worldid, obj1id]), xquat_in[worldid, obj2id])       <L 3062>
                        var_133 = wp::address(var_xquat_in, var_0, var_99);
                        var_135 = wp::load(var_133);
                        var_134 = quat_inv_0(var_135);
                        var_136 = wp::address(var_xquat_in, var_0, var_102);
                        var_138 = wp::load(var_136);
                        var_137 = mul_quat_0(var_134, var_138);
                        // data[6] = relquat[0]                                                   <L 3063>
                        var_140 = wp::extract(var_137, var_139);
                        wp::assign_inplace(var_14, var_141, var_140);
                        // data[7] = relquat[1]                                                   <L 3064>
                        var_143 = wp::extract(var_137, var_142);
                        wp::assign_inplace(var_14, var_144, var_143);
                        // data[8] = relquat[2]                                                   <L 3065>
                        var_146 = wp::extract(var_137, var_145);
                        wp::assign_inplace(var_14, var_147, var_146);
                        // data[9] = relquat[3]                                                   <L 3066>
                        var_149 = wp::extract(var_137, var_148);
                        wp::assign_inplace(var_14, var_150, var_149);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 3067>
                        wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    var_151 = wp::where(var_84, var_85, var_81);
                }
            }
        }
        var_152 = wp::where(var_19, var_25, var_99);
        var_153 = wp::where(var_19, var_28, var_102);
        var_154 = wp::where(var_19, var_36, var_123);
        var_155 = wp::where(var_19, var_41, var_115);
        var_156 = wp::where(var_19, var_49, var_110);
}



extern "C" __global__ void _compute_eq_data0_d31148f7_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data_out,
    wp::array_t<wp::int32> adj_eq_type,
    wp::array_t<wp::int32> adj_eq_obj1id,
    wp::array_t<wp::int32> adj_eq_obj2id,
    wp::array_t<wp::int32> adj_eq_objtype,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> adj_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_xmat_in,
    wp::array_t<wp::vec_t<11, wp::float32>> adj_eq_data_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<11, wp::float32>* var_13;
        wp::vec_t<11, wp::float32> var_14;
        wp::vec_t<11, wp::float32> var_15;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        bool var_19;
        const wp::int32 var_20 = 1;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::mat_t<3, 3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::mat_t<3, 3, wp::float32>* var_43;
        wp::mat_t<3, 3, wp::float32> var_44;
        wp::mat_t<3, 3, wp::float32> var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        const wp::int32 var_50 = 0;
        wp::float32 var_51;
        const wp::int32 var_52 = 3;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        const wp::int32 var_55 = 4;
        const wp::int32 var_56 = 2;
        wp::float32 var_57;
        const wp::int32 var_58 = 5;
        const wp::int32 var_59 = 6;
        const wp::int32 var_60 = 6;
        wp::int32 var_61;
        bool var_62;
        const wp::float32 var_63 = 0.0;
        wp::vec_t<11, wp::float32> var_64;
        const wp::int32 var_65 = 1;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        bool var_68;
        const wp::int32 var_69 = 1;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        bool var_72;
        const wp::int32 var_73 = 6;
        wp::float32 var_74;
        const wp::int32 var_75 = 7;
        wp::float32 var_76;
        const wp::int32 var_77 = 8;
        wp::float32 var_78;
        const wp::int32 var_79 = 9;
        wp::float32 var_80;
        wp::quat_t<wp::float32> var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        wp::quat_t<wp::float32> var_85;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        const wp::int32 var_88 = 6;
        const wp::int32 var_89 = 1;
        wp::float32 var_90;
        const wp::int32 var_91 = 7;
        const wp::int32 var_92 = 2;
        wp::float32 var_93;
        const wp::int32 var_94 = 8;
        const wp::int32 var_95 = 3;
        wp::float32 var_96;
        const wp::int32 var_97 = 9;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        const wp::int32 var_106 = 1;
        wp::float32 var_107;
        const wp::int32 var_108 = 2;
        wp::float32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32>* var_111;
        wp::mat_t<3, 3, wp::float32>* var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::mat_t<3, 3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::mat_t<3, 3, wp::float32>* var_117;
        wp::mat_t<3, 3, wp::float32> var_118;
        wp::mat_t<3, 3, wp::float32> var_119;
        wp::vec_t<3, wp::float32>* var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::vec_t<3, wp::float32> var_123;
        const wp::int32 var_124 = 0;
        wp::float32 var_125;
        const wp::int32 var_126 = 3;
        const wp::int32 var_127 = 1;
        wp::float32 var_128;
        const wp::int32 var_129 = 4;
        const wp::int32 var_130 = 2;
        wp::float32 var_131;
        const wp::int32 var_132 = 5;
        wp::quat_t<wp::float32>* var_133;
        wp::quat_t<wp::float32> var_134;
        wp::quat_t<wp::float32> var_135;
        wp::quat_t<wp::float32>* var_136;
        wp::quat_t<wp::float32> var_137;
        wp::quat_t<wp::float32> var_138;
        const wp::int32 var_139 = 0;
        wp::float32 var_140;
        const wp::int32 var_141 = 6;
        const wp::int32 var_142 = 1;
        wp::float32 var_143;
        const wp::int32 var_144 = 7;
        const wp::int32 var_145 = 2;
        wp::float32 var_146;
        const wp::int32 var_147 = 8;
        const wp::int32 var_148 = 3;
        wp::float32 var_149;
        const wp::int32 var_150 = 9;
        wp::quat_t<wp::float32> var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::vec_t<11, wp::float32> adj_13 = {};
        wp::vec_t<11, wp::float32> adj_14 = {};
        wp::vec_t<11, wp::float32> adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        bool adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        bool adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::mat_t<3, 3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::mat_t<3, 3, wp::float32> adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::mat_t<3, 3, wp::float32> adj_43 = {};
        wp::mat_t<3, 3, wp::float32> adj_44 = {};
        wp::mat_t<3, 3, wp::float32> adj_45 = {};
        wp::vec_t<3, wp::float32> adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::vec_t<3, wp::float32> adj_49 = {};
        wp::int32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        bool adj_62 = {};
        wp::float32 adj_63 = {};
        wp::vec_t<11, wp::float32> adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        bool adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::int32 adj_71 = {};
        bool adj_72 = {};
        wp::int32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::quat_t<wp::float32> adj_81 = {};
        wp::float32 adj_82 = {};
        wp::float32 adj_83 = {};
        bool adj_84 = {};
        wp::quat_t<wp::float32> adj_85 = {};
        wp::int32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::float32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::int32 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::int32 adj_98 = {};
        wp::int32 adj_99 = {};
        wp::int32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::float32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::int32 adj_108 = {};
        wp::float32 adj_109 = {};
        wp::vec_t<3, wp::float32> adj_110 = {};
        wp::vec_t<3, wp::float32> adj_111 = {};
        wp::mat_t<3, 3, wp::float32> adj_112 = {};
        wp::vec_t<3, wp::float32> adj_113 = {};
        wp::mat_t<3, 3, wp::float32> adj_114 = {};
        wp::vec_t<3, wp::float32> adj_115 = {};
        wp::vec_t<3, wp::float32> adj_116 = {};
        wp::mat_t<3, 3, wp::float32> adj_117 = {};
        wp::mat_t<3, 3, wp::float32> adj_118 = {};
        wp::mat_t<3, 3, wp::float32> adj_119 = {};
        wp::vec_t<3, wp::float32> adj_120 = {};
        wp::vec_t<3, wp::float32> adj_121 = {};
        wp::vec_t<3, wp::float32> adj_122 = {};
        wp::vec_t<3, wp::float32> adj_123 = {};
        wp::int32 adj_124 = {};
        wp::float32 adj_125 = {};
        wp::int32 adj_126 = {};
        wp::int32 adj_127 = {};
        wp::float32 adj_128 = {};
        wp::int32 adj_129 = {};
        wp::int32 adj_130 = {};
        wp::float32 adj_131 = {};
        wp::int32 adj_132 = {};
        wp::quat_t<wp::float32> adj_133 = {};
        wp::quat_t<wp::float32> adj_134 = {};
        wp::quat_t<wp::float32> adj_135 = {};
        wp::quat_t<wp::float32> adj_136 = {};
        wp::quat_t<wp::float32> adj_137 = {};
        wp::quat_t<wp::float32> adj_138 = {};
        wp::int32 adj_139 = {};
        wp::float32 adj_140 = {};
        wp::int32 adj_141 = {};
        wp::int32 adj_142 = {};
        wp::float32 adj_143 = {};
        wp::int32 adj_144 = {};
        wp::int32 adj_145 = {};
        wp::float32 adj_146 = {};
        wp::int32 adj_147 = {};
        wp::int32 adj_148 = {};
        wp::float32 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::quat_t<wp::float32> adj_151 = {};
        wp::int32 adj_152 = {};
        wp::int32 adj_153 = {};
        wp::vec_t<3, wp::float32> adj_154 = {};
        wp::vec_t<3, wp::float32> adj_155 = {};
        wp::vec_t<3, wp::float32> adj_156 = {};
        //---------
        // forward
        // def _compute_eq_data0(                                                                 <L 2994>
        // worldid, eqid = wp.tid()                                                               <L 3011>
        builtin_tid2d(var_0, var_1);
        // eq_data_id = worldid % eq_data_out.shape[0]                                            <L 3012>
        var_2 = &(var_eq_data_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // eqtype = eq_type[eqid]                                                                 <L 3014>
        var_7 = wp::address(var_eq_type, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // objtype = eq_objtype[eqid]                                                             <L 3015>
        var_10 = wp::address(var_eq_objtype, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // data = eq_data_out[eq_data_id, eqid]                                                   <L 3016>
        var_13 = wp::address(var_eq_data_out, var_6, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if eqtype == int(types.EqType.CONNECT.value):                                          <L 3018>
        var_18 = wp::int(var_17);
        var_19 = (var_8 == var_18);
        if (var_19) {
            // if objtype == int(types.ObjType.BODY.value):                                       <L 3019>
            var_22 = wp::int(var_21);
            var_23 = (var_11 == var_22);
            if (var_23) {
                // obj1id = eq_obj1id[eqid]                                                       <L 3020>
                var_24 = wp::address(var_eq_obj1id, var_1);
                var_26 = wp::load(var_24);
                var_25 = wp::copy(var_26);
                // obj2id = eq_obj2id[eqid]                                                       <L 3021>
                var_27 = wp::address(var_eq_obj2id, var_1);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // anchor1 = wp.vec3(data[0], data[1], data[2])                                   <L 3024>
                var_31 = wp::extract(var_14, var_30);
                var_33 = wp::extract(var_14, var_32);
                var_35 = wp::extract(var_14, var_34);
                var_36 = wp::vec_t<3, wp::float32>(var_31, var_33, var_35);
                // pos = xpos_in[worldid, obj1id] + xmat_in[worldid, obj1id] @ anchor1            <L 3025>
                var_37 = wp::address(var_xpos_in, var_0, var_25);
                var_38 = wp::address(var_xmat_in, var_0, var_25);
                var_40 = wp::load(var_38);
                var_39 = wp::mul(var_40, var_36);
                var_42 = wp::load(var_37);
                var_41 = wp::add(var_42, var_39);
                // anchor2 = wp.transpose(xmat_in[worldid, obj2id]) @ (pos - xpos_in[worldid, obj2id])       <L 3028>
                var_43 = wp::address(var_xmat_in, var_0, var_28);
                var_45 = wp::load(var_43);
                var_44 = wp::transpose(var_45);
                var_46 = wp::address(var_xpos_in, var_0, var_28);
                var_48 = wp::load(var_46);
                var_47 = wp::sub(var_41, var_48);
                var_49 = wp::mul(var_44, var_47);
                // data[3] = anchor2[0]                                                           <L 3029>
                var_51 = wp::extract(var_49, var_50);
                wp::assign_inplace(var_14, var_52, var_51);
                // data[4] = anchor2[1]                                                           <L 3030>
                var_54 = wp::extract(var_49, var_53);
                wp::assign_inplace(var_14, var_55, var_54);
                // data[5] = anchor2[2]                                                           <L 3031>
                var_57 = wp::extract(var_49, var_56);
                wp::assign_inplace(var_14, var_58, var_57);
                // eq_data_out[eq_data_id, eqid] = data                                           <L 3032>
                // wp::array_store(var_eq_data_out, var_6, var_1, var_14);
            }
            if (!var_23) {
                // elif objtype == int(types.ObjType.SITE.value):                                 <L 3033>
                var_61 = wp::int(var_60);
                var_62 = (var_11 == var_61);
                if (var_62) {
                    // eq_data_out[eq_data_id, eqid] = types.vec11(0.0)                           <L 3035>
                    var_64 = wp::vec_t<11, wp::float32>(var_63);
                    // wp::array_store(var_eq_data_out, var_6, var_1, var_64);
                }
            }
        }
        if (!var_19) {
            // elif eqtype == int(types.EqType.WELD.value):                                       <L 3036>
            var_67 = wp::int(var_66);
            var_68 = (var_8 == var_67);
            if (var_68) {
                // if objtype == int(types.ObjType.BODY.value):                                   <L 3037>
                var_71 = wp::int(var_70);
                var_72 = (var_11 == var_71);
                if (var_72) {
                    // quat = wp.quat(data[6], data[7], data[8], data[9])                         <L 3038>
                    var_74 = wp::extract(var_14, var_73);
                    var_76 = wp::extract(var_14, var_75);
                    var_78 = wp::extract(var_14, var_77);
                    var_80 = wp::extract(var_14, var_79);
                    var_81 = wp::quat_t<wp::float32>(var_74, var_76, var_78, var_80);
                    // if wp.length_sq(quat) > 0.0:                                               <L 3039>
                    var_82 = wp::length_sq(var_81);
                    var_84 = (var_82 > var_83);
                    if (var_84) {
                        // quat = wp.normalize(quat)                                              <L 3041>
                        var_85 = wp::normalize(var_81);
                        // data[6] = quat[0]                                                      <L 3042>
                        var_87 = wp::extract(var_85, var_86);
                        wp::assign_inplace(var_14, var_88, var_87);
                        // data[7] = quat[1]                                                      <L 3043>
                        var_90 = wp::extract(var_85, var_89);
                        wp::assign_inplace(var_14, var_91, var_90);
                        // data[8] = quat[2]                                                      <L 3044>
                        var_93 = wp::extract(var_85, var_92);
                        wp::assign_inplace(var_14, var_94, var_93);
                        // data[9] = quat[3]                                                      <L 3045>
                        var_96 = wp::extract(var_85, var_95);
                        wp::assign_inplace(var_14, var_97, var_96);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 3046>
                        // wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    if (!var_84) {
                        // obj1id = eq_obj1id[eqid]                                               <L 3048>
                        var_98 = wp::address(var_eq_obj1id, var_1);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // obj2id = eq_obj2id[eqid]                                               <L 3049>
                        var_101 = wp::address(var_eq_obj2id, var_1);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // anchor2 = wp.vec3(data[0], data[1], data[2])                           <L 3052>
                        var_105 = wp::extract(var_14, var_104);
                        var_107 = wp::extract(var_14, var_106);
                        var_109 = wp::extract(var_14, var_108);
                        var_110 = wp::vec_t<3, wp::float32>(var_105, var_107, var_109);
                        // pos = xpos_in[worldid, obj2id] + xmat_in[worldid, obj2id] @ anchor2       <L 3053>
                        var_111 = wp::address(var_xpos_in, var_0, var_102);
                        var_112 = wp::address(var_xmat_in, var_0, var_102);
                        var_114 = wp::load(var_112);
                        var_113 = wp::mul(var_114, var_110);
                        var_116 = wp::load(var_111);
                        var_115 = wp::add(var_116, var_113);
                        // anchor1 = wp.transpose(xmat_in[worldid, obj1id]) @ (pos - xpos_in[worldid, obj1id])       <L 3056>
                        var_117 = wp::address(var_xmat_in, var_0, var_99);
                        var_119 = wp::load(var_117);
                        var_118 = wp::transpose(var_119);
                        var_120 = wp::address(var_xpos_in, var_0, var_99);
                        var_122 = wp::load(var_120);
                        var_121 = wp::sub(var_115, var_122);
                        var_123 = wp::mul(var_118, var_121);
                        // data[3] = anchor1[0]                                                   <L 3057>
                        var_125 = wp::extract(var_123, var_124);
                        wp::assign_inplace(var_14, var_126, var_125);
                        // data[4] = anchor1[1]                                                   <L 3058>
                        var_128 = wp::extract(var_123, var_127);
                        wp::assign_inplace(var_14, var_129, var_128);
                        // data[5] = anchor1[2]                                                   <L 3059>
                        var_131 = wp::extract(var_123, var_130);
                        wp::assign_inplace(var_14, var_132, var_131);
                        // relquat = mjmath.mul_quat(mjmath.quat_inv(xquat_in[worldid, obj1id]), xquat_in[worldid, obj2id])       <L 3062>
                        var_133 = wp::address(var_xquat_in, var_0, var_99);
                        var_135 = wp::load(var_133);
                        var_134 = quat_inv_0(var_135);
                        var_136 = wp::address(var_xquat_in, var_0, var_102);
                        var_138 = wp::load(var_136);
                        var_137 = mul_quat_0(var_134, var_138);
                        // data[6] = relquat[0]                                                   <L 3063>
                        var_140 = wp::extract(var_137, var_139);
                        wp::assign_inplace(var_14, var_141, var_140);
                        // data[7] = relquat[1]                                                   <L 3064>
                        var_143 = wp::extract(var_137, var_142);
                        wp::assign_inplace(var_14, var_144, var_143);
                        // data[8] = relquat[2]                                                   <L 3065>
                        var_146 = wp::extract(var_137, var_145);
                        wp::assign_inplace(var_14, var_147, var_146);
                        // data[9] = relquat[3]                                                   <L 3066>
                        var_149 = wp::extract(var_137, var_148);
                        wp::assign_inplace(var_14, var_150, var_149);
                        // eq_data_out[eq_data_id, eqid] = data                                   <L 3067>
                        // wp::array_store(var_eq_data_out, var_6, var_1, var_14);
                    }
                    var_151 = wp::where(var_84, var_85, var_81);
                }
            }
        }
        var_152 = wp::where(var_19, var_25, var_99);
        var_153 = wp::where(var_19, var_28, var_102);
        var_154 = wp::where(var_19, var_36, var_123);
        var_155 = wp::where(var_19, var_41, var_115);
        var_156 = wp::where(var_19, var_49, var_110);
        //---------
        // reverse
        wp::adj_where(var_19, var_49, var_110, adj_19, adj_49, adj_110, adj_156);
        wp::adj_where(var_19, var_41, var_115, adj_19, adj_41, adj_115, adj_155);
        wp::adj_where(var_19, var_36, var_123, adj_19, adj_36, adj_123, adj_154);
        wp::adj_where(var_19, var_28, var_102, adj_19, adj_28, adj_102, adj_153);
        wp::adj_where(var_19, var_25, var_99, adj_19, adj_25, adj_99, adj_152);
        if (!var_19) {
            if (var_68) {
                if (var_72) {
                    wp::adj_where(var_84, var_85, var_81, adj_84, adj_85, adj_81, adj_151);
                    if (!var_84) {
                        wp::adj_array_store(var_eq_data_out, var_6, var_1, var_14, adj_eq_data_out, adj_6, adj_1, adj_14);
                        // adj: eq_data_out[eq_data_id, eqid] = data                              <L 3067>
                        wp::adj_assign_inplace(var_14, var_150, var_149, adj_14, adj_150, adj_149);
                        wp::adj_extract(var_137, var_148, adj_137, adj_148, adj_149);
                        // adj: data[9] = relquat[3]                                              <L 3066>
                        wp::adj_assign_inplace(var_14, var_147, var_146, adj_14, adj_147, adj_146);
                        wp::adj_extract(var_137, var_145, adj_137, adj_145, adj_146);
                        // adj: data[8] = relquat[2]                                              <L 3065>
                        wp::adj_assign_inplace(var_14, var_144, var_143, adj_14, adj_144, adj_143);
                        wp::adj_extract(var_137, var_142, adj_137, adj_142, adj_143);
                        // adj: data[7] = relquat[1]                                              <L 3064>
                        wp::adj_assign_inplace(var_14, var_141, var_140, adj_14, adj_141, adj_140);
                        wp::adj_extract(var_137, var_139, adj_137, adj_139, adj_140);
                        // adj: data[6] = relquat[0]                                              <L 3063>
                        adj_mul_quat_0(var_134, var_138, adj_134, adj_136, adj_137);
                        wp::adj_address(var_xquat_in, var_0, var_102, adj_xquat_in, adj_0, adj_102, adj_136);
                        adj_quat_inv_0(var_135, adj_133, adj_134);
                        wp::adj_address(var_xquat_in, var_0, var_99, adj_xquat_in, adj_0, adj_99, adj_133);
                        // adj: relquat = mjmath.mul_quat(mjmath.quat_inv(xquat_in[worldid, obj1id]), xquat_in[worldid, obj2id])  <L 3062>
                        wp::adj_assign_inplace(var_14, var_132, var_131, adj_14, adj_132, adj_131);
                        wp::adj_extract(var_123, var_130, adj_123, adj_130, adj_131);
                        // adj: data[5] = anchor1[2]                                              <L 3059>
                        wp::adj_assign_inplace(var_14, var_129, var_128, adj_14, adj_129, adj_128);
                        wp::adj_extract(var_123, var_127, adj_123, adj_127, adj_128);
                        // adj: data[4] = anchor1[1]                                              <L 3058>
                        wp::adj_assign_inplace(var_14, var_126, var_125, adj_14, adj_126, adj_125);
                        wp::adj_extract(var_123, var_124, adj_123, adj_124, adj_125);
                        // adj: data[3] = anchor1[0]                                              <L 3057>
                        wp::adj_mul(var_118, var_121, adj_118, adj_121, adj_123);
                        wp::adj_sub(var_115, var_122, adj_115, adj_120, adj_121);
                        wp::adj_address(var_xpos_in, var_0, var_99, adj_xpos_in, adj_0, adj_99, adj_120);
                        wp::adj_transpose(var_119, adj_117, adj_118);
                        wp::adj_address(var_xmat_in, var_0, var_99, adj_xmat_in, adj_0, adj_99, adj_117);
                        // adj: anchor1 = wp.transpose(xmat_in[worldid, obj1id]) @ (pos - xpos_in[worldid, obj1id])  <L 3056>
                        wp::adj_add(var_116, var_113, adj_111, adj_113, adj_115);
                        wp::adj_mul(var_114, var_110, adj_112, adj_110, adj_113);
                        wp::adj_address(var_xmat_in, var_0, var_102, adj_xmat_in, adj_0, adj_102, adj_112);
                        wp::adj_address(var_xpos_in, var_0, var_102, adj_xpos_in, adj_0, adj_102, adj_111);
                        // adj: pos = xpos_in[worldid, obj2id] + xmat_in[worldid, obj2id] @ anchor2  <L 3053>
                        wp::adj_vec_t(var_105, var_107, var_109, adj_105, adj_107, adj_109, adj_110);
                        wp::adj_extract(var_14, var_108, adj_14, adj_108, adj_109);
                        wp::adj_extract(var_14, var_106, adj_14, adj_106, adj_107);
                        wp::adj_extract(var_14, var_104, adj_14, adj_104, adj_105);
                        // adj: anchor2 = wp.vec3(data[0], data[1], data[2])                      <L 3052>
                        wp::adj_copy(var_103, adj_101, adj_102);
                        wp::adj_address(var_eq_obj2id, var_1, adj_eq_obj2id, adj_1, adj_101);
                        // adj: obj2id = eq_obj2id[eqid]                                          <L 3049>
                        wp::adj_copy(var_100, adj_98, adj_99);
                        wp::adj_address(var_eq_obj1id, var_1, adj_eq_obj1id, adj_1, adj_98);
                        // adj: obj1id = eq_obj1id[eqid]                                          <L 3048>
                    }
                    if (var_84) {
                        wp::adj_array_store(var_eq_data_out, var_6, var_1, var_14, adj_eq_data_out, adj_6, adj_1, adj_14);
                        // adj: eq_data_out[eq_data_id, eqid] = data                              <L 3046>
                        wp::adj_assign_inplace(var_14, var_97, var_96, adj_14, adj_97, adj_96);
                        wp::adj_extract(var_85, var_95, adj_85, adj_95, adj_96);
                        // adj: data[9] = quat[3]                                                 <L 3045>
                        wp::adj_assign_inplace(var_14, var_94, var_93, adj_14, adj_94, adj_93);
                        wp::adj_extract(var_85, var_92, adj_85, adj_92, adj_93);
                        // adj: data[8] = quat[2]                                                 <L 3044>
                        wp::adj_assign_inplace(var_14, var_91, var_90, adj_14, adj_91, adj_90);
                        wp::adj_extract(var_85, var_89, adj_85, adj_89, adj_90);
                        // adj: data[7] = quat[1]                                                 <L 3043>
                        wp::adj_assign_inplace(var_14, var_88, var_87, adj_14, adj_88, adj_87);
                        wp::adj_extract(var_85, var_86, adj_85, adj_86, adj_87);
                        // adj: data[6] = quat[0]                                                 <L 3042>
                        wp::adj_normalize(var_81, adj_81, adj_85);
                        // adj: quat = wp.normalize(quat)                                         <L 3041>
                    }
                    wp::adj_length_sq(var_81, adj_81, adj_82);
                    // adj: if wp.length_sq(quat) > 0.0:                                          <L 3039>
                    wp::adj_quat_t(var_74, var_76, var_78, var_80, adj_74, adj_76, adj_78, adj_80, adj_81);
                    wp::adj_extract(var_14, var_79, adj_14, adj_79, adj_80);
                    wp::adj_extract(var_14, var_77, adj_14, adj_77, adj_78);
                    wp::adj_extract(var_14, var_75, adj_14, adj_75, adj_76);
                    wp::adj_extract(var_14, var_73, adj_14, adj_73, adj_74);
                    // adj: quat = wp.quat(data[6], data[7], data[8], data[9])                    <L 3038>
                }
                // adj: if objtype == int(types.ObjType.BODY.value):                              <L 3037>
            }
            // adj: elif eqtype == int(types.EqType.WELD.value):                                  <L 3036>
        }
        if (var_19) {
            if (!var_23) {
                if (var_62) {
                    wp::adj_array_store(var_eq_data_out, var_6, var_1, var_64, adj_eq_data_out, adj_6, adj_1, adj_64);
                    wp::adj_vec_t(var_63, adj_63, adj_64);
                    // adj: eq_data_out[eq_data_id, eqid] = types.vec11(0.0)                      <L 3035>
                }
                // adj: elif objtype == int(types.ObjType.SITE.value):                            <L 3033>
            }
            if (var_23) {
                wp::adj_array_store(var_eq_data_out, var_6, var_1, var_14, adj_eq_data_out, adj_6, adj_1, adj_14);
                // adj: eq_data_out[eq_data_id, eqid] = data                                      <L 3032>
                wp::adj_assign_inplace(var_14, var_58, var_57, adj_14, adj_58, adj_57);
                wp::adj_extract(var_49, var_56, adj_49, adj_56, adj_57);
                // adj: data[5] = anchor2[2]                                                      <L 3031>
                wp::adj_assign_inplace(var_14, var_55, var_54, adj_14, adj_55, adj_54);
                wp::adj_extract(var_49, var_53, adj_49, adj_53, adj_54);
                // adj: data[4] = anchor2[1]                                                      <L 3030>
                wp::adj_assign_inplace(var_14, var_52, var_51, adj_14, adj_52, adj_51);
                wp::adj_extract(var_49, var_50, adj_49, adj_50, adj_51);
                // adj: data[3] = anchor2[0]                                                      <L 3029>
                wp::adj_mul(var_44, var_47, adj_44, adj_47, adj_49);
                wp::adj_sub(var_41, var_48, adj_41, adj_46, adj_47);
                wp::adj_address(var_xpos_in, var_0, var_28, adj_xpos_in, adj_0, adj_28, adj_46);
                wp::adj_transpose(var_45, adj_43, adj_44);
                wp::adj_address(var_xmat_in, var_0, var_28, adj_xmat_in, adj_0, adj_28, adj_43);
                // adj: anchor2 = wp.transpose(xmat_in[worldid, obj2id]) @ (pos - xpos_in[worldid, obj2id])  <L 3028>
                wp::adj_add(var_42, var_39, adj_37, adj_39, adj_41);
                wp::adj_mul(var_40, var_36, adj_38, adj_36, adj_39);
                wp::adj_address(var_xmat_in, var_0, var_25, adj_xmat_in, adj_0, adj_25, adj_38);
                wp::adj_address(var_xpos_in, var_0, var_25, adj_xpos_in, adj_0, adj_25, adj_37);
                // adj: pos = xpos_in[worldid, obj1id] + xmat_in[worldid, obj1id] @ anchor1       <L 3025>
                wp::adj_vec_t(var_31, var_33, var_35, adj_31, adj_33, adj_35, adj_36);
                wp::adj_extract(var_14, var_34, adj_14, adj_34, adj_35);
                wp::adj_extract(var_14, var_32, adj_14, adj_32, adj_33);
                wp::adj_extract(var_14, var_30, adj_14, adj_30, adj_31);
                // adj: anchor1 = wp.vec3(data[0], data[1], data[2])                              <L 3024>
                wp::adj_copy(var_29, adj_27, adj_28);
                wp::adj_address(var_eq_obj2id, var_1, adj_eq_obj2id, adj_1, adj_27);
                // adj: obj2id = eq_obj2id[eqid]                                                  <L 3021>
                wp::adj_copy(var_26, adj_24, adj_25);
                wp::adj_address(var_eq_obj1id, var_1, adj_eq_obj1id, adj_1, adj_24);
                // adj: obj1id = eq_obj1id[eqid]                                                  <L 3020>
            }
            // adj: if objtype == int(types.ObjType.BODY.value):                                  <L 3019>
        }
        // adj: if eqtype == int(types.EqType.CONNECT.value):                                     <L 3018>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_eq_data_out, var_6, var_1, adj_eq_data_out, adj_6, adj_1, adj_13);
        // adj: data = eq_data_out[eq_data_id, eqid]                                              <L 3016>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_eq_objtype, var_1, adj_eq_objtype, adj_1, adj_10);
        // adj: objtype = eq_objtype[eqid]                                                        <L 3015>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_eq_type, var_1, adj_eq_type, adj_1, adj_7);
        // adj: eqtype = eq_type[eqid]                                                            <L 3014>
        wp::adj_mod(var_0, var_4, adj_0, adj_4, adj_6);
        adj_eq_data_out.shape = adj_2;
        // adj: eq_data_id = worldid % eq_data_out.shape[0]                                       <L 3012>
        // adj: worldid, eqid = wp.tid()                                                          <L 3011>
        // adj: def _compute_eq_data0(                                                            <L 2994>
        return;
}



extern "C" __global__ void _compute_body_A_diag_entry_d84c280f_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::float32> var_body_jac_row_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_body_A_diag_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def _compute_body_A_diag_entry(                                                        <L 3236>
        // worldid = wp.tid()                                                                     <L 3244>
        var_0 = builtin_tid1d();
        // body_A_diag_id = worldid % body_A_diag_out.shape[0]                                    <L 3245>
        var_1 = &(var_body_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 3247>
        var_7 = wp::float(var_6);
        // for i in range(nv):                                                                    <L 3248>
        var_8 = wp::range(var_nv);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // dot_prod += body_jac_row_in[worldid, i] * result_vec_in[worldid, i]                <L 3249>
            var_10 = wp::address(var_body_jac_row_in, var_0, var_9);
            var_11 = wp::address(var_result_vec_in, var_0, var_9);
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = wp::mul(var_13, var_14);
            var_15 = wp::add(var_7, var_12);
            wp::assign(var_7, var_15);
            goto start_for_0;
        end_for_0:;
        // body_A_diag_out[body_A_diag_id, bodyid_target, row_idx] = dot_prod                     <L 3250>
        wp::array_store(var_body_A_diag_out, var_5, var_bodyid_target, var_row_idx, var_7);
}



extern "C" __global__ void _compute_body_A_diag_entry_d84c280f_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::int32 var_bodyid_target,
    wp::int32 var_row_idx,
    wp::array_t<wp::float32> var_body_jac_row_in,
    wp::array_t<wp::float32> var_result_vec_in,
    wp::array_t<wp::float32> var_body_A_diag_out,
    wp::int32 adj_nv,
    wp::int32 adj_bodyid_target,
    wp::int32 adj_row_idx,
    wp::array_t<wp::float32> adj_body_jac_row_in,
    wp::array_t<wp::float32> adj_result_vec_in,
    wp::array_t<wp::float32> adj_body_A_diag_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::shape_t adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::range_t adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        //---------
        // forward
        // def _compute_body_A_diag_entry(                                                        <L 3236>
        // worldid = wp.tid()                                                                     <L 3244>
        var_0 = builtin_tid1d();
        // body_A_diag_id = worldid % body_A_diag_out.shape[0]                                    <L 3245>
        var_1 = &(var_body_A_diag_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        var_5 = wp::mod(var_0, var_3);
        // dot_prod = float(0.0)                                                                  <L 3247>
        var_7 = wp::float(var_6);
        // for i in range(nv):                                                                    <L 3248>
        var_8 = wp::range(var_nv);
        // body_A_diag_out[body_A_diag_id, bodyid_target, row_idx] = dot_prod                     <L 3250>
        // wp::array_store(var_body_A_diag_out, var_5, var_bodyid_target, var_row_idx, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_body_A_diag_out, var_5, var_bodyid_target, var_row_idx, var_7, adj_body_A_diag_out, adj_5, adj_bodyid_target, adj_row_idx, adj_7);
        // adj: body_A_diag_out[body_A_diag_id, bodyid_target, row_idx] = dot_prod                <L 3250>
        var_8 = wp::iter_reverse(var_8);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
        	adj_10 = {};
        	adj_11 = {};
        	adj_12 = {};
        	adj_13 = {};
        	adj_14 = {};
        	adj_15 = {};
            // dot_prod += body_jac_row_in[worldid, i] * result_vec_in[worldid, i]                <L 3249>
            var_10 = wp::address(var_body_jac_row_in, var_0, var_9);
            var_11 = wp::address(var_result_vec_in, var_0, var_9);
            var_13 = wp::load(var_10);
            var_14 = wp::load(var_11);
            var_12 = wp::mul(var_13, var_14);
            var_15 = wp::add(var_7, var_12);
            wp::assign(var_7, var_15);
            wp::adj_assign(var_7, var_15, adj_7, adj_15);
            wp::adj_add(var_7, var_12, adj_7, adj_12, adj_15);
            wp::adj_mul(var_13, var_14, adj_10, adj_11, adj_12);
            wp::adj_address(var_result_vec_in, var_0, var_9, adj_result_vec_in, adj_0, adj_9, adj_11);
            wp::adj_address(var_body_jac_row_in, var_0, var_9, adj_body_jac_row_in, adj_0, adj_9, adj_10);
            // adj: dot_prod += body_jac_row_in[worldid, i] * result_vec_in[worldid, i]           <L 3249>
        	goto start_for_0;
        end_for_0:;
        // adj: for i in range(nv):                                                               <L 3248>
        wp::adj_float(var_6, adj_6, adj_7);
        // adj: dot_prod = float(0.0)                                                             <L 3247>
        wp::adj_mod(var_0, var_3, adj_0, adj_3, adj_5);
        adj_body_A_diag_out.shape = adj_1;
        // adj: body_A_diag_id = worldid % body_A_diag_out.shape[0]                               <L 3245>
        // adj: worldid = wp.tid()                                                                <L 3244>
        // adj: def _compute_body_A_diag_entry(                                                   <L 3236>
        return;
}



extern "C" __global__ void _compute_dof_M0_eaae0df3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::float32> var_dof_armature,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_dof_M0_out)
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
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::vec_t<10, wp::float32>* var_13;
        wp::vec_t<6, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<10, wp::float32> var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32>* var_18;
        wp::float32 var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _compute_dof_M0(                                                                   <L 3422>
        // worldid, dofid = wp.tid()                                                              <L 3429>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 3430>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // armature = dof_armature[worldid % dof_armature.shape[0], dofid]                        <L 3431>
        var_5 = &(var_dof_armature.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_dof_armature, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // buf = mjmath.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])               <L 3432>
        var_13 = wp::address(var_crb_in, var_0, var_3);
        var_14 = wp::address(var_cdof_in, var_0, var_1);
        var_16 = wp::load(var_13);
        var_17 = wp::load(var_14);
        var_15 = inert_vec_0(var_16, var_17);
        // dof_M0_out[worldid, dofid] = armature + wp.dot(cdof_in[worldid, dofid], buf)           <L 3433>
        var_18 = wp::address(var_cdof_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::dot(var_20, var_15);
        var_21 = wp::add(var_11, var_19);
        wp::array_store(var_dof_M0_out, var_0, var_1, var_21);
}



extern "C" __global__ void _compute_dof_M0_eaae0df3_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::float32> var_dof_armature,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_dof_M0_out,
    wp::array_t<wp::int32> adj_dof_bodyid,
    wp::array_t<wp::float32> adj_dof_armature,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> adj_crb_in,
    wp::array_t<wp::float32> adj_dof_M0_out)
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
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::vec_t<10, wp::float32>* var_13;
        wp::vec_t<6, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<10, wp::float32> var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32>* var_18;
        wp::float32 var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::float32 var_21;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::shape_t adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::shape_t adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::vec_t<10, wp::float32> adj_13 = {};
        wp::vec_t<6, wp::float32> adj_14 = {};
        wp::vec_t<6, wp::float32> adj_15 = {};
        wp::vec_t<10, wp::float32> adj_16 = {};
        wp::vec_t<6, wp::float32> adj_17 = {};
        wp::vec_t<6, wp::float32> adj_18 = {};
        wp::float32 adj_19 = {};
        wp::vec_t<6, wp::float32> adj_20 = {};
        wp::float32 adj_21 = {};
        //---------
        // forward
        // def _compute_dof_M0(                                                                   <L 3422>
        // worldid, dofid = wp.tid()                                                              <L 3429>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 3430>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // armature = dof_armature[worldid % dof_armature.shape[0], dofid]                        <L 3431>
        var_5 = &(var_dof_armature.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_dof_armature, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // buf = mjmath.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])               <L 3432>
        var_13 = wp::address(var_crb_in, var_0, var_3);
        var_14 = wp::address(var_cdof_in, var_0, var_1);
        var_16 = wp::load(var_13);
        var_17 = wp::load(var_14);
        var_15 = inert_vec_0(var_16, var_17);
        // dof_M0_out[worldid, dofid] = armature + wp.dot(cdof_in[worldid, dofid], buf)           <L 3433>
        var_18 = wp::address(var_cdof_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::dot(var_20, var_15);
        var_21 = wp::add(var_11, var_19);
        // wp::array_store(var_dof_M0_out, var_0, var_1, var_21);
        //---------
        // reverse
        wp::adj_array_store(var_dof_M0_out, var_0, var_1, var_21, adj_dof_M0_out, adj_0, adj_1, adj_21);
        wp::adj_add(var_11, var_19, adj_11, adj_19, adj_21);
        wp::adj_dot(var_20, var_15, adj_18, adj_15, adj_19);
        wp::adj_address(var_cdof_in, var_0, var_1, adj_cdof_in, adj_0, adj_1, adj_18);
        // adj: dof_M0_out[worldid, dofid] = armature + wp.dot(cdof_in[worldid, dofid], buf)      <L 3433>
        adj_inert_vec_0(var_16, var_17, adj_13, adj_14, adj_15);
        wp::adj_address(var_cdof_in, var_0, var_1, adj_cdof_in, adj_0, adj_1, adj_14);
        wp::adj_address(var_crb_in, var_0, var_3, adj_crb_in, adj_0, adj_3, adj_13);
        // adj: buf = mjmath.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])          <L 3432>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_dof_armature, var_9, var_1, adj_dof_armature, adj_9, adj_1, adj_10);
        wp::adj_mod(var_0, var_7, adj_0, adj_7, adj_9);
        adj_dof_armature.shape = adj_5;
        // adj: armature = dof_armature[worldid % dof_armature.shape[0], dofid]                   <L 3431>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_dof_bodyid, var_1, adj_dof_bodyid, adj_1, adj_2);
        // adj: bodyid = dof_bodyid[dofid]                                                        <L 3430>
        // adj: worldid, dofid = wp.tid()                                                         <L 3429>
        // adj: def _compute_dof_M0(                                                              <L 3422>
        return;
}



extern "C" __global__ void _set_unit_vector_f74e5cf9_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid_target,
    wp::array_t<wp::float32> var_unit_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::float32 var_8 = 1.0;
        const wp::float32 var_9 = 0.0;
        //---------
        // forward
        // def _set_unit_vector(                                                                  <L 3108>
        // worldid = wp.tid()                                                                     <L 3112>
        var_0 = builtin_tid1d();
        // nv = unit_vec_out.shape[1]                                                             <L 3113>
        var_1 = &(var_unit_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 3114>
        var_5 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // if i == dofid_target:                                                              <L 3115>
            var_7 = (var_6 == var_dofid_target);
            if (var_7) {
                // unit_vec_out[worldid, i] = 1.0                                                 <L 3116>
                wp::array_store(var_unit_vec_out, var_0, var_6, var_8);
            }
            if (!var_7) {
                // unit_vec_out[worldid, i] = 0.0                                                 <L 3118>
                wp::array_store(var_unit_vec_out, var_0, var_6, var_9);
            }
            goto start_for_0;
        end_for_0:;
}



extern "C" __global__ void _set_unit_vector_f74e5cf9_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_dofid_target,
    wp::array_t<wp::float32> var_unit_vec_out,
    wp::int32 adj_dofid_target,
    wp::array_t<wp::float32> adj_unit_vec_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::shape_t* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::shape_t var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        bool var_7;
        const wp::float32 var_8 = 1.0;
        const wp::float32 var_9 = 0.0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::shape_t adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::range_t adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        //---------
        // forward
        // def _set_unit_vector(                                                                  <L 3108>
        // worldid = wp.tid()                                                                     <L 3112>
        var_0 = builtin_tid1d();
        // nv = unit_vec_out.shape[1]                                                             <L 3113>
        var_1 = &(var_unit_vec_out.shape);
        var_4 = wp::load(var_1);
        var_3 = wp::extract(var_4, var_2);
        // for i in range(nv):                                                                    <L 3114>
        var_5 = wp::range(var_3);
        //---------
        // reverse
        var_5 = wp::iter_reverse(var_5);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
        	adj_7 = {};
        	adj_8 = {};
        	adj_9 = {};
            // if i == dofid_target:                                                              <L 3115>
            var_7 = (var_6 == var_dofid_target);
            if (var_7) {
                // unit_vec_out[worldid, i] = 1.0                                                 <L 3116>
                // wp::array_store(var_unit_vec_out, var_0, var_6, var_8);
            }
            if (!var_7) {
                // unit_vec_out[worldid, i] = 0.0                                                 <L 3118>
                // wp::array_store(var_unit_vec_out, var_0, var_6, var_9);
            }
            if (!var_7) {
                wp::adj_array_store(var_unit_vec_out, var_0, var_6, var_9, adj_unit_vec_out, adj_0, adj_6, adj_9);
                // adj: unit_vec_out[worldid, i] = 0.0                                            <L 3118>
            }
            if (var_7) {
                wp::adj_array_store(var_unit_vec_out, var_0, var_6, var_8, adj_unit_vec_out, adj_0, adj_6, adj_8);
                // adj: unit_vec_out[worldid, i] = 1.0                                            <L 3116>
            }
            // adj: if i == dofid_target:                                                         <L 3115>
        	goto start_for_0;
        end_for_0:;
        // adj: for i in range(nv):                                                               <L 3114>
        adj_unit_vec_out.shape = adj_1;
        // adj: nv = unit_vec_out.shape[1]                                                        <L 3113>
        // adj: worldid = wp.tid()                                                                <L 3112>
        // adj: def _set_unit_vector(                                                             <L 3108>
        return;
}

