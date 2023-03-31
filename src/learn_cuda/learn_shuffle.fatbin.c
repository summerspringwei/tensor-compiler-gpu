#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000014a8,0x0000004001010002,0x00000000000010c0\n"
".quad 0x0000000000000000,0x0000003400010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007500be0002,0x0000000000000000,0x0000000000000000,0x0000000000000d00\n"
".quad 0x0000004000340534,0x0001000f00400000,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x2e747865742e006f,0x6365765f74736574,0x2e766e2e00726f74,0x7365742e6f666e69\n"
".quad 0x726f746365765f74,0x6168732e766e2e00,0x747365742e646572,0x00726f746365765f\n"
".quad 0x736e6f632e766e2e,0x65742e30746e6174,0x6f746365765f7473,0x2e747865742e0072\n"
".quad 0x7568735f74736574,0x766e2e00656c6666,0x65742e6f666e692e,0x66667568735f7473\n"
".quad 0x732e766e2e00656c,0x65742e6465726168,0x66667568735f7473,0x632e766e2e00656c\n"
".quad 0x30746e6174736e6f,0x68735f747365742e,0x742e00656c666675,0x747365742e747865\n"
".quad 0x697469646e6f635f,0x692e766e2e006e6f,0x747365742e6f666e,0x697469646e6f635f\n"
".quad 0x732e766e2e006e6f,0x65742e6465726168,0x69646e6f635f7473,0x766e2e006e6f6974\n"
".quad 0x6e6174736e6f632e,0x5f747365742e3074,0x6f697469646e6f63,0x65722e766e2e006e\n"
".quad 0x6e6f697463612e6c,0x72747368732e0000,0x7274732e00626174,0x6d79732e00626174\n"
".quad 0x6d79732e00626174,0x646e68735f626174,0x6e692e766e2e0078,0x5f74736574006f66\n"
".quad 0x2e00726f74636576,0x7365742e74786574,0x726f746365765f74,0x666e692e766e2e00\n"
".quad 0x765f747365742e6f,0x6e2e00726f746365,0x6465726168732e76,0x65765f747365742e\n"
".quad 0x766e2e00726f7463,0x6e6174736e6f632e,0x5f747365742e3074,0x5f00726f74636576\n"
".quad 0x6574006d61726170,0x66667568735f7473,0x747865742e00656c,0x68735f747365742e\n"
".quad 0x6e2e00656c666675,0x742e6f666e692e76,0x667568735f747365,0x2e766e2e00656c66\n"
".quad 0x742e646572616873,0x667568735f747365,0x2e766e2e00656c66,0x746e6174736e6f63\n"
".quad 0x735f747365742e30,0x7400656c66667568,0x646e6f635f747365,0x742e006e6f697469\n"
".quad 0x747365742e747865,0x697469646e6f635f,0x692e766e2e006e6f,0x747365742e6f666e\n"
".quad 0x697469646e6f635f,0x732e766e2e006e6f,0x65742e6465726168,0x69646e6f635f7473\n"
".quad 0x766e2e006e6f6974,0x6e6174736e6f632e,0x5f747365742e3074,0x6f697469646e6f63\n"
".quad 0x65722e766e2e006e,0x6e6f697463612e6c,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x000c00030000003e,0x0000000000000000\n"
".quad 0x0000000000000000,0x000900030000007c,0x0000000000000000,0x0000000000000000\n"
".quad 0x000d0003000000aa,0x0000000000000000,0x0000000000000000,0x000a0003000000eb\n"
".quad 0x0000000000000000,0x0000000000000000,0x000e000300000115,0x0000000000000000\n"
".quad 0x0000000000000000,0x000b00030000015c,0x0000000000000000,0x0000000000000000\n"
".quad 0x0008000300000179,0x0000000000000000,0x0000000000000000,0x000c101200000032\n"
".quad 0x0000000000000000,0x0000000000000080,0x000d10120000009d,0x0000000000000000\n"
".quad 0x0000000000000100,0x000e101200000106,0x0000000000000000,0x00000000000000c0\n"
".quad 0x0000000a00082f04,0x0008230400000004,0x000000000000000a,0x0000000a00081204\n"
".quad 0x0008110400000000,0x000000000000000a,0x0000000900082f04,0x0008230400000006\n"
".quad 0x0000000000000009,0x0000000900081204,0x0008110400000000,0x0000000000000009\n"
".quad 0x0000000800082f04,0x0008230400000006,0x0000000000000008,0x0000000800081204\n"
".quad 0x0008110400000000,0x0000000000000008,0x0000007500043704,0x00002a0100003001\n"
".quad 0x0000000200080a04,0x0010190300100140,0x00000000000c1704,0x0021f00000080001\n"
".quad 0x00000000000c1704,0x0021f00000000000,0x00041c0400ff1b03,0x000c050400000070\n"
".quad 0x0000000100000080,0x0004370400000001,0x0000300100000075,0x00080a0400002a01\n"
".quad 0x0010014000000004,0x000c170400101903,0x0008000100000000,0x000c17040021f000\n"
".quad 0x0000000000000000,0x00ff1b030021f000,0x0000007000042804,0x000000b800041c04\n"
".quad 0x00000080000c0504,0x0000000100000001,0x0000007500043704,0x00002a0100003001\n"
".quad 0x0000000600080a04,0x0010190300100140,0x00000000000c1704,0x0021f00000080001\n"
".quad 0x00000000000c1704,0x0021f00000000000,0x00041c0400ff1b03,0x000c050400000078\n"
".quad 0x0000000100000080,0x0000000000000001,0x000000000000004b,0x222f0a1008020200\n"
".quad 0x0000000008000000,0x0000000008080000,0x0000000008100000,0x0000000008180000\n"
".quad 0x0000000008200000,0x0000000008280000,0x0000000008300000,0x0000000008380000\n"
".quad 0x0000000008000001,0x0000000008080001,0x0000000008100001,0x0000000008180001\n"
".quad 0x0000000008200001,0x0000000008280001,0x0000000008300001,0x0000000008380001\n"
".quad 0x0000000008000002,0x0000000008080002,0x0000000008100002,0x0000000008180002\n"
".quad 0x0000000008200002,0x0000000008280002,0x0000000008300002,0x0000000008380002\n"
".quad 0x0000002c14000000,0x000000000c000009,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x083fc400e3e007f6,0x4c98078000870001\n"
".quad 0xf0c8000002170000,0x3848000000270004,0x001fc840fec007f5,0x3828000001e70000\n"
".quad 0x4c10800005070402,0x4c10080005170003,0x001fdc00fcc007b1,0xeed4200000070202\n"
".quad 0x4c10800005270404,0x4c10080005370005,0x001ffc00ffe107f1,0xeedc200000070402\n"
".quad 0xe30000000007000f,0xe2400fffff87000f,0x001c7c00fe0007f6,0x4c98078000870001\n"
".quad 0x010000000207f004,0xf0c8000002170005,0x001fc800fec20ff1,0x3828000001e70500\n"
".quad 0x4c18810005070502,0x4c10080005170003,0x001fd441fe2000b4,0xeed4200000070200\n"
".quad 0x5bdf7f8010470402,0x3669038000370407,0x001ffc20e22007f9,0x3829000000170202\n"
".quad 0xef17007ca0270003,0x5c98078000270004,0x081fd800ffa00ff0,0x5c58000000070300\n"
".quad 0xe2400ffffb80000f,0x4bd7810005270502,0x001ffc00fea007f2,0x1a177f8005370503\n"
".quad 0xeedc200000070200,0xe30000000007000f,0x001f8000fc0007ff,0xe2400fffff07000f\n"
".quad 0x50b0000000070f00,0x50b0000000070f00,0x001f8000fc0007e0,0x50b0000000070f00\n"
".quad 0x50b0000000070f00,0x50b0000000070f00,0x083fc400e3e007f6,0x4c98078000870001\n"
".quad 0xf0c8000002170002,0x0400000000170200,0x001fc400fe8207f1,0x3828000001e70203\n"
".quad 0x4c18810005270202,0x366a038000170007,0x001fc800fd6007f1,0x010fffffffe7f000\n"
".quad 0x4c10080005370303,0x38a0040000170000,0x001ffc01fe20071d,0x5cb8000000072a00\n"
".quad 0xeedc200000070200,0xe30000000007000f,0x001f8000fc0007ff,0xe2400fffff07000f\n"
".quad 0x50b0000000070f00,0x50b0000000070f00,0x001f8000fc0007e0,0x50b0000000070f00\n"
".quad 0x50b0000000070f00,0x50b0000000070f00,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000300000001,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000040,0x0000000000000159,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x000000030000000b,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000199,0x0000000000000188,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x0000000200000013,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000328,0x0000000000000108,0x0000000800000002\n"
".quad 0x0000000000000008,0x0000000000000018,0x7000000000000029,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000430,0x0000000000000090,0x0000000000000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x7000000000000044,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000004c0,0x000000000000005c,0x0000000c00000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x700000000000009d,0x0000000000000000\n"
".quad 0x0000000000000000,0x000000000000051c,0x0000000000000064,0x0000000d00000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x70000000000000fb,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000580,0x000000000000005c,0x0000000e00000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x7000000b0000014a,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000005e0,0x00000000000000e0,0x0000000000000000\n"
".quad 0x0000000000000008,0x0000000000000008,0x0000000100000070,0x0000000000000002\n"
".quad 0x0000000000000000,0x00000000000006c0,0x0000000000000150,0x0000000c00000000\n"
".quad 0x0000000000000004,0x0000000000000000,0x00000001000000cb,0x0000000000000002\n"
".quad 0x0000000000000000,0x0000000000000810,0x0000000000000150,0x0000000d00000000\n"
".quad 0x0000000000000004,0x0000000000000000,0x000000010000012d,0x0000000000000002\n"
".quad 0x0000000000000000,0x0000000000000960,0x0000000000000150,0x0000000e00000000\n"
".quad 0x0000000000000004,0x0000000000000000,0x0000000100000032,0x0000000000000006\n"
".quad 0x0000000000000000,0x0000000000000ac0,0x0000000000000080,0x0600000800000003\n"
".quad 0x0000000000000020,0x0000000000000000,0x000000010000008a,0x0000000000000006\n"
".quad 0x0000000000000000,0x0000000000000b40,0x0000000000000100,0x0600000900000003\n"
".quad 0x0000000000000020,0x0000000000000000,0x00000001000000e6,0x0000000000000006\n"
".quad 0x0000000000000000,0x0000000000000c40,0x00000000000000c0,0x0400000a00000003\n"
".quad 0x0000000000000020,0x0000000000000000,0x0000004801010001,0x0000000000000360\n"
".quad 0x000000400000035d,0x0000003400070007,0x0000000000000000,0x0000000000002011\n"
".quad 0x0000000000000000,0x0000000000000792,0x0000000000000000,0x762e1cf200010a13\n"
".quad 0x37206e6f69737265,0x677261742e0a372e,0x32355f6d73207465,0x7365726464612e0a\n"
".quad 0x3620657a69735f73,0x6973691afb002f34,0x746e652e20656c62,0x5f74736574207972\n"
".quad 0x6f697469646e6f63,0x617261702e0a286e,0x001c3436752e206d,0x2c305f3f001a5f11\n"
".quad 0x290a311bf30f0024,0x69746e78616d2e0a,0x31202c3832312064,0x722e0a7b0a31202c\n"
".quad 0x646572702e206765,0x123b3e323c702520,0x6625203233666700,0x7236001162100011\n"
".quad 0x343600f20011343c,0x3b3e353c64722520,0x220084646c0a0a0a,0x202c314f0017752e\n"
".quad 0x3b5d02f403008a5b,0x6f742e617476630a,0x336c61626f6c672e,0x3b7100392c322100\n"
".quad 0x006f752e766f6d0a,0x64697425202c31e3,0x84646e610a3b782e,0x31a2001a2c322200\n"
".quad 0x652e707465730a3b,0x1e2c317032001971,0x2f706c2300190100,0xf0322d202c335100\n"
".quad 0x30007c3170253100,0x732e2000e06e722e,0x33b3003666110020,0x69772e6c756d0a3b\n"
".quad 0x67336423007b6564,0x6464610a3b348200,0xab2c342600a5732e,0xc574730a3b335400\n"
".quad 0x00215b1001310000,0x0a3b9f00552c5d20,0xe17d0a0a3b746572,0x66667568737f0401\n"
".quad 0x001a030001df656c,0xdb0f00220e01dd0e,0x381601db341c1901,0x313c3d00f10101ca\n"
".quad 0x01dd30312f01dc32,0x00ac0b01dd331402,0x2b321f002b3b5d2d,0x1601e75d31280300\n"
".quad 0x7534019f0001e734,0x1f019d0101853436,0x0f018d0205023234,0x0c003735110001bc\n"
".quad 0x1100352c362601bc,0x254101bc0700ab35,0x9b361901c12c3766,0x4157202c3131b300\n"
".quad 0x46026f5a535f5052,0x2000026f732e746c,0x25400a3b3208f700,0x2420617262203170\n"
".quad 0x335f3142425f5f4c,0x202c384900460a3b,0x2d202c398402d933,0x3221002f0a0a3b31\n"
".quad 0xbe00018a02001d3a,0x68730a3b37667300,0x23006b3614002f72,0x1400170100d23133\n"
".quad 0x2f36722532001737,0x001d0102f3731400,0x2e6c6668d3030900,0x776f642e636e7973\n"
".quad 0x257c30317300616e,0x002f02006b2c3270,0x873b392600053810,0x303122001c661100\n"
".quad 0x0005030123050071,0x010e6714010e3513,0x1400e80800a33315,0x3370254049035e31\n"
".quad 0x332f00fe32170121,0x01ac37110401e13a,0x02956c1400c73210,0xdd321902192c3823\n"
".quad 0x381f00312c392601,0x01575d3923010399,0x6365766f0b03990f,0x1902000398726f74\n"
".quad 0x0f00210e03970e00,0x0e05055f0f100396,0x0a0b055f381f0383,0x9f0a0903810f0096\n"
".quad 0x7333140605860f00,0x05a50f0903730f01,0x88311f0203880f05,0x3507038831161c03\n"
".quad 0x351f009e2c372600, 0x50057937190101e0, 0x000000000a0a7d0a\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[663];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif
