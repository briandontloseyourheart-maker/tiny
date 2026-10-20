#ifndef RIEMANN_LLF_COMPRESSIBLE_EULER_H_
#define RIEMANN_LLF_COMPRESSIBLE_EULER_H_

#include <cmath>

#include "Conservative2Primitive_CompressibleEuler.H"
#include "Primitive2State_CompressibleEuler.H"
#include "Conservative2Flux_CompressibleEuler.H"

// 单气体 Euler 方程的 LLF 界面通量
// ID_Dimension：0=x，1=y，2=z
// 要求：0 <= ID_Dimension < AMREX_SPACEDIM
// 左右状态必须具有正密度、正压力，且 gamma > 1
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Riemann_LLF(
    // 输出
    amrex::GpuArray<amrex::Real, NCONS>& Flux_Riemann,//当前界面的 LLF 数值通量
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& Vel_Riemann,//左右速度的算术平均
    amrex::Real& cmax,//左右“声速加完整速度模长”的最大值

    // 输入
    amrex::GpuArray<amrex::Real, NCONS> const& ConservativeL,
    amrex::GpuArray<amrex::Real, NCONS> const& ConservativeR,
    amrex::Real const gamma,
    int const ID_Dimension)
{
    // 1. 恢复左右原始状态
    amrex::GpuArray<amrex::Real, NPRIM> PrimitiveL{};
    amrex::GpuArray<amrex::Real, NPRIM> PrimitiveR{};

    amrex::Real eL;
    amrex::Real eR;

    Conservative2Primitive(
        PrimitiveL, eL, ConservativeL, gamma
    );

    Conservative2Primitive(
        PrimitiveR, eR, ConservativeR, gamma
    );

    // 2. 计算左右物理通量
    amrex::GpuArray<amrex::Real, NCONS> FluxL{};
    amrex::GpuArray<amrex::Real, NCONS> FluxR{};

    Conservative2Flux(
        FluxL, ConservativeL, gamma, ID_Dimension
    );

    Conservative2Flux(
        FluxR, ConservativeR, gamma, ID_Dimension
    );

    // 3. 从原始数组取出密度、速度、压力
    amrex::Real rhoL;
    amrex::Real rhoR;
    amrex::Real pL;
    amrex::Real pR;

    AMREX_D_TERM(
    amrex::Real uL;,
    amrex::Real vL;,
    amrex::Real wL;);

    AMREX_D_TERM(
    amrex::Real uR;,
    amrex::Real vR;,
    amrex::Real wR;);

    Primitive2State(
        rhoL,
        AMREX_D_DECL(uL, vL, wL),
        pL,
        PrimitiveL
    );

    Primitive2State(
        rhoR,
        AMREX_D_DECL(uR, vR, wR),
        pR,
        PrimitiveR
    );

    // 4. 理想气体声速
    amrex::Real cL = std::sqrt(gamma * pL / rhoL);
    amrex::Real cR = std::sqrt(gamma * pR / rhoR);

    // 5. 当前方向的左右速度
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> velocityL{
        AMREX_D_DECL(uL, vL, wL)
    };

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> velocityR{
        AMREX_D_DECL(uR, vR, wR)
    };

    amrex::Real normalL = velocityL[ID_Dimension];
    amrex::Real normalR = velocityR[ID_Dimension];

    amrex::Real absNormalL =//绝对值
        (normalL >= 0.0) ? normalL : -normalL;

    amrex::Real absNormalR =
        (normalR >= 0.0) ? normalR : -normalR;

    amrex::Real speedL = absNormalL + cL;
    amrex::Real speedR = absNormalR + cR;

    amrex::Real Lambda =
        (speedL > speedR) ? speedL : speedR;

    // 6. 输出左右速度的算术平均
    amrex::Real velocitySquaredL = 0.0;
    amrex::Real velocitySquaredR = 0.0;

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        Vel_Riemann[d] =
            0.5 * (velocityL[d] + velocityR[d]);

        velocitySquaredL += velocityL[d] * velocityL[d];
        velocitySquaredR += velocityR[d] * velocityR[d];
    }

    // 7. LLF 数值通量：所有守恒分量共用同一个 Lambda
    for (int n = 0; n < NCONS; ++n) {
        Flux_Riemann[n] =
            0.5 * (
                FluxL[n] + FluxR[n]
                - Lambda * (
                    ConservativeR[n] - ConservativeL[n]
                )
            );
    }

    // 8. 声速加完整速度模长的最大值
    amrex::Real totalSpeedL =
        cL + std::sqrt(velocitySquaredL);

    amrex::Real totalSpeedR =
        cR + std::sqrt(velocitySquaredR);

    cmax =
        (totalSpeedL > totalSpeedR)
        ? totalSpeedL
        : totalSpeedR;
}

#endif