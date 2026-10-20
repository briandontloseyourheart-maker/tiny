#ifndef VARIABLES_COMPRESSIBLE_EULER_H_
#define VARIABLES_COMPRESSIBLE_EULER_H_

#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>
#include <string>

// 守恒变量名称：顺序必须与 StateIndex 一致
inline amrex::Vector<std::string> vnamesC = {
    "rho",
    AMREX_D_DECL("rho_u", "rho_v", "rho_w"),
    "rho_E"
};

// 原始变量名称：顺序必须与 PrimitiveIndex 一致
inline amrex::Vector<std::string> vnamesP = {
    "rho",
    AMREX_D_DECL("u", "v", "w"),
    "P"
};

// 守恒变量分量编号
enum StateIndex {
    URHO = 0,

    AMREX_D_DECL(
    UMX = 1,
    UMY = 2,
    UMZ = 3),

    UENE = 1 + AMREX_SPACEDIM
};

// 原始变量分量编号
enum PrimitiveIndex {
    VRHO = 0,

    AMREX_D_DECL(
    VU = 1,
    VV = 2,
    VW = 3),

    VP = 1 + AMREX_SPACEDIM
};

// 分量数量
inline constexpr int NCONS = 2 + AMREX_SPACEDIM;
inline constexpr int NPRIM = 2 + AMREX_SPACEDIM;

#endif