#ifndef CONSERVATIVE2FLUX_COMPRESSIBLE_EULER_H_
#define CONSERVATIVE2FLUX_COMPRESSIBLE_EULER_H_

#include "Conservative2Primitive_CompressibleEuler.H"
#include "Primitive2State_CompressibleEuler.H"
#include "Conservative2State_CompressibleEuler.H"
#include "Component2Vector_CompressibleEuler.H"

// 根据守恒状态计算指定方向的物理通量
// ID_Dimension：0=x，1=y，2=z
// 要求：0 <= ID_Dimension < AMREX_SPACEDIM
// 输入状态要求正密度、正内能，且 gamma > 1
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Conservative2Flux(
    // 输出：物理通量数组
    amrex::GpuArray<amrex::Real, NCONS>& Flux,

    // 输入：守恒状态、比热比、方向
    amrex::GpuArray<amrex::Real, NCONS> const& Conservative,
    amrex::Real const gamma,
    int const ID_Dimension)//计算哪个方向的通量
{
    // 1. 守恒数组转换为原始数组
    amrex::GpuArray<amrex::Real, NPRIM> Primitive{};
    amrex::Real e;

    Conservative2Primitive(
        Primitive,
        e,
        Conservative,
        gamma
    );

    // 2. 取出密度、速度、压力
    amrex::Real rho;
    amrex::Real p;

    AMREX_D_TERM(
    amrex::Real u;,
    amrex::Real v;,
    amrex::Real w;);

    Primitive2State(
        rho,
        AMREX_D_DECL(u, v, w),
        p,
        Primitive
    );//拆数组

    // 3. //读取守恒数组，取出动量密度和总能量密度
    amrex::Real rho_E;

    AMREX_D_TERM(
    amrex::Real rho_u;,
    amrex::Real rho_v;,
    amrex::Real rho_w;);

    Conservative2State(
        rho,
        AMREX_D_DECL(rho_u, rho_v, rho_w),
        rho_E,
        Conservative
    );

    // 4. 选择当前方向的速度
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> velocity{
        AMREX_D_DECL(u, v, w)
    };

    amrex::Real normal_velocity = velocity[ID_Dimension];

    // 5. 装入质量、动量输运、总能量通量
    Component2Vector(
        Flux,
        rho * normal_velocity,

        AMREX_D_DECL(
        rho_u * normal_velocity,
        rho_v * normal_velocity,
        rho_w * normal_velocity),

        (rho_E + p) * normal_velocity
    );//此时压力项没加上

    // 6. 压力只加在当前方向的法向动量通量上
    Flux[UMX + ID_Dimension] += p;
}

#endif