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



extern "C" __global__ void unpack_rgb_kernel_56f2a71c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::uint32> var_packed,
    wp::array_t<wp::int32> var_rgb_adr,
    wp::int32 var_camera_index,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rgb_out)
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
        const wp::int32 var_3 = 2;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 2;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::uint32* var_16;
        wp::uint32 var_17;
        wp::uint32 var_18;
        const wp::int32 var_19 = 255;
        wp::uint32 var_20;
        wp::uint32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.00392156862745098;
        wp::float32 var_24;
        const wp::int32 var_25 = 8;
        wp::uint32 var_26;
        wp::uint32 var_27;
        const wp::int32 var_28 = 255;
        wp::uint32 var_29;
        wp::uint32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.00392156862745098;
        wp::float32 var_33;
        const wp::int32 var_34 = 16;
        wp::uint32 var_35;
        wp::uint32 var_36;
        const wp::int32 var_37 = 255;
        wp::uint32 var_38;
        wp::uint32 var_39;
        wp::float32 var_40;
        const wp::float32 var_41 = 0.00392156862745098;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        //---------
        // forward
        // def unpack_rgb_kernel(                                                                 <L 135>
        // worldid, pixelid = wp.tid()                                                            <L 144>
        builtin_tid2d(var_0, var_1);
        // xid = pixelid % rgb_out.shape[2]                                                       <L 146>
        var_2 = &(var_rgb_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_1, var_4);
        // yid = pixelid // rgb_out.shape[2]                                                      <L 147>
        var_7 = &(var_rgb_out.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::floordiv(var_1, var_9);
        // rgb_adr_offset = rgb_adr[camera_index]                                                 <L 149>
        var_12 = wp::address(var_rgb_adr, var_camera_index);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // val = packed[worldid, rgb_adr_offset + pixelid]                                        <L 150>
        var_15 = wp::add(var_13, var_1);
        var_16 = wp::address(var_packed, var_0, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // b = wp.float32(val & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)                         <L 151>
        var_20 = wp::uint32(var_19);
        var_21 = wp::bit_and(var_17, var_20);
        var_22 = wp::float32(var_21);
        var_24 = wp::mul(var_22, var_23);
        // g = wp.float32((val >> wp.uint32(8)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)       <L 152>
        var_26 = wp::uint32(var_25);
        var_27 = wp::rshift(var_17, var_26);
        var_29 = wp::uint32(var_28);
        var_30 = wp::bit_and(var_27, var_29);
        var_31 = wp::float32(var_30);
        var_33 = wp::mul(var_31, var_32);
        // r = wp.float32((val >> wp.uint32(16)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)       <L 153>
        var_35 = wp::uint32(var_34);
        var_36 = wp::rshift(var_17, var_35);
        var_38 = wp::uint32(var_37);
        var_39 = wp::bit_and(var_36, var_38);
        var_40 = wp::float32(var_39);
        var_42 = wp::mul(var_40, var_41);
        // rgb_out[worldid, yid, xid] = wp.vec3(r, g, b)                                          <L 154>
        var_43 = wp::vec_t<3, wp::float32>(var_42, var_33, var_24);
        wp::array_store(var_rgb_out, var_0, var_11, var_6, var_43);
}



extern "C" __global__ void _extract_seg_kernel_61a3c915_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<2, wp::int32>> var_seg_data,
    wp::array_t<wp::int32> var_seg_adr,
    wp::int32 var_camera_index,
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 2;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 2;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<2, wp::int32>* var_16;
        wp::vec_t<2, wp::int32> var_17;
        //---------
        // forward
        // def _extract_seg_kernel(                                                               <L 212>
        // worldid, pixelid = wp.tid()                                                            <L 221>
        builtin_tid2d(var_0, var_1);
        // xid = pixelid % seg_out.shape[2]                                                       <L 222>
        var_2 = &(var_seg_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_1, var_4);
        // yid = pixelid // seg_out.shape[2]                                                      <L 223>
        var_7 = &(var_seg_out.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::floordiv(var_1, var_9);
        // seg_adr_offset = seg_adr[camera_index]                                                 <L 225>
        var_12 = wp::address(var_seg_adr, var_camera_index);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // seg_out[worldid, yid, xid] = seg_data[worldid, seg_adr_offset + pixelid]               <L 226>
        var_15 = wp::add(var_13, var_1);
        var_16 = wp::address(var_seg_data, var_0, var_15);
        var_17 = wp::load(var_16);
        wp::array_store(var_seg_out, var_0, var_11, var_6, var_17);
}



extern "C" __global__ void extract_depth_kernel_c75e5a3d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_depth_data,
    wp::array_t<wp::int32> var_depth_adr,
    wp::int32 var_camera_index,
    wp::float32 var_depth_scale,
    wp::array_t<wp::float32> var_depth_out)
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
        const wp::int32 var_3 = 2;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 2;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 1.0;
        wp::float32 var_22;
        //---------
        // forward
        // def extract_depth_kernel(                                                              <L 158>
        // worldid, pixelid = wp.tid()                                                            <L 168>
        builtin_tid2d(var_0, var_1);
        // xid = pixelid % depth_out.shape[2]                                                     <L 169>
        var_2 = &(var_depth_out.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_1, var_4);
        // yid = pixelid // depth_out.shape[2]                                                    <L 170>
        var_7 = &(var_depth_out.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::floordiv(var_1, var_9);
        // depth_adr_offset = depth_adr[camera_index]                                             <L 172>
        var_12 = wp::address(var_depth_adr, var_camera_index);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // val = depth_data[worldid, depth_adr_offset + pixelid]                                  <L 173>
        var_15 = wp::add(var_13, var_1);
        var_16 = wp::address(var_depth_data, var_0, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // depth_out[worldid, yid, xid] = wp.clamp(val / depth_scale, 0.0, 1.0)                   <L 174>
        var_19 = wp::div(var_17, var_depth_scale);
        var_22 = wp::clamp(var_19, var_20, var_21);
        wp::array_store(var_depth_out, var_0, var_11, var_6, var_22);
}



extern "C" __global__ void _convert_texture_data_ec489c73_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_width,
    wp::int32 var_adr,
    wp::int32 var_nc,
    wp::array_t<wp::uint8> var_tex_data_in,
    wp::array_t<wp::float32> var_tex_data_out)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::uint8* var_10;
        wp::uint8 var_11;
        wp::uint8 var_12;
        const wp::int32 var_13 = 0;
        wp::uint8 var_14;
        wp::uint8 var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::uint8* var_20;
        wp::uint8 var_21;
        wp::uint8 var_22;
        const wp::int32 var_23 = 0;
        wp::uint8 var_24;
        wp::uint8 var_25;
        const wp::int32 var_26 = 2;
        bool var_27;
        const wp::int32 var_28 = 2;
        wp::int32 var_29;
        wp::uint8* var_30;
        wp::uint8 var_31;
        wp::uint8 var_32;
        const wp::int32 var_33 = 0;
        wp::uint8 var_34;
        wp::uint8 var_35;
        const wp::int32 var_36 = 255;
        wp::uint8 var_37;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.00392156862745098;
        wp::float32 var_40;
        const wp::int32 var_41 = 0;
        wp::float32 var_42;
        const wp::float32 var_43 = 0.00392156862745098;
        wp::float32 var_44;
        const wp::int32 var_45 = 1;
        wp::float32 var_46;
        const wp::float32 var_47 = 0.00392156862745098;
        wp::float32 var_48;
        const wp::int32 var_49 = 2;
        wp::float32 var_50;
        const wp::float32 var_51 = 0.00392156862745098;
        wp::float32 var_52;
        const wp::int32 var_53 = 3;
        //---------
        // forward
        // def _convert_texture_data(                                                             <L 26>
        // x, y = wp.tid()                                                                        <L 36>
        builtin_tid2d(var_0, var_1);
        // offset = adr + (y * width + x) * nc                                                    <L 37>
        var_2 = wp::mul(var_1, var_width);
        var_3 = wp::add(var_2, var_0);
        var_4 = wp::mul(var_3, var_nc);
        var_5 = wp::add(var_adr, var_4);
        // r = tex_data_in[offset + 0] if nc > 0 else wp.uint8(0)                                 <L 38>
        var_7 = (var_nc > var_6);
        if (var_7) {
            var_9 = wp::add(var_5, var_8);
            var_10 = wp::address(var_tex_data_in, var_9);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
        }
        if (!var_7) {
            var_14 = wp::uint8(var_13);
        }
        var_15 = wp::where(var_7, var_11, var_14);
        // g = tex_data_in[offset + 1] if nc > 1 else wp.uint8(0)                                 <L 39>
        var_17 = (var_nc > var_16);
        if (var_17) {
            var_19 = wp::add(var_5, var_18);
            var_20 = wp::address(var_tex_data_in, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
        }
        if (!var_17) {
            var_24 = wp::uint8(var_23);
        }
        var_25 = wp::where(var_17, var_21, var_24);
        // b = tex_data_in[offset + 2] if nc > 2 else wp.uint8(0)                                 <L 40>
        var_27 = (var_nc > var_26);
        if (var_27) {
            var_29 = wp::add(var_5, var_28);
            var_30 = wp::address(var_tex_data_in, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
        }
        if (!var_27) {
            var_34 = wp::uint8(var_33);
        }
        var_35 = wp::where(var_27, var_31, var_34);
        // a = wp.uint8(255)                                                                      <L 41>
        var_37 = wp::uint8(var_36);
        // tex_data_out[y, x, 0] = float(r) * wp.static(1.0 / 255.0)                              <L 43>
        var_38 = wp::float(var_15);
        var_40 = wp::mul(var_38, var_39);
        wp::array_store(var_tex_data_out, var_1, var_0, var_41, var_40);
        // tex_data_out[y, x, 1] = float(g) * wp.static(1.0 / 255.0)                              <L 44>
        var_42 = wp::float(var_25);
        var_44 = wp::mul(var_42, var_43);
        wp::array_store(var_tex_data_out, var_1, var_0, var_45, var_44);
        // tex_data_out[y, x, 2] = float(b) * wp.static(1.0 / 255.0)                              <L 45>
        var_46 = wp::float(var_35);
        var_48 = wp::mul(var_46, var_47);
        wp::array_store(var_tex_data_out, var_1, var_0, var_49, var_48);
        // tex_data_out[y, x, 3] = float(a) * wp.static(1.0 / 255.0)                              <L 46>
        var_50 = wp::float(var_37);
        var_52 = wp::mul(var_50, var_51);
        wp::array_store(var_tex_data_out, var_1, var_0, var_53, var_52);
}

