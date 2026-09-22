// AvbdSolverVk — Vulkan compute backend for the AVBD vertex-block-update
// pipeline. Implements the same `cloth::AvbdSolver` API as the Metal
// backend in AvbdSolver.mm, so Simulation.cpp needs no per-backend code.
//
// The kernels are the same Lean-generated Slang as the Metal path; only
// the target differs (slangc -target spirv instead of -target metal ->
// .air -> .metallib). The Lean codegen already annotates every buffer
// with [[vk::binding(n, 0)]], so descriptor layouts come straight from
// the shaders via AvbdKernelTable.inc, which is generated from
// slangc -reflection-json. Adding or reordering a buffer in Lean
// regenerates the table; there is no hand-maintained binding list.
//
// Two things differ materially from Metal and are the source of nearly
// all porting risk:
//
//   1. Barriers. AvbdSolver.mm encodes the whole color loop into one
//      MTLComputeCommandEncoder and leans on Metal's implicit serial
//      dispatch ordering -- it contains no explicit barrier anywhere.
//      Vulkan guarantees nothing, so we emit a pipeline barrier after
//      every dispatch. Missing one yields plausible-but-wrong cloth
//      rather than a crash, so this is deliberately unconditional.
//
//   2. Ragged tails. Metal's dispatchThreads: takes an exact thread
//      count and handles a non-uniform final threadgroup. vkCmdDispatch
//      takes group counts, so ceil(N/64) groups run up to 63 extra
//      lanes -- and none of the 13 AVBD kernels bounds-check tid. We
//      fix this host-side with sentinel padding rather than patching
//      Lean (whose emitted text is pinned by native_decide proofs):
//      every buffer is allocated to roundUp(n + 1, 64) elements, index
//      `n` is a dummy slot, and each color's vertPerm range is padded
//      to a multiple of 64 with that sentinel. Overrun lanes then
//      read and write only the dummy slot.
//
// float3 is padded to 16 bytes, matching both SPIR-V's ArrayStride 16
// and Metal's bytesVec3Padded, so the host packing is identical to the
// Metal backend's. (The CPU backend in the RISC-V guest uses tight
// 12-byte vec3 -- its layout is NOT interchangeable with this one.)

#include "AvbdSolver.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "AvbdCsr.h"
#include "AvbdKernelTable.inc"

namespace cloth {
namespace {

constexpr uint32_t kThreadsPerGroup = 64;

// Every buffer is sized so that a ragged-tail lane (up to 63 past the
// real element count) lands on a valid slot, and so that index `n`
// itself exists as the sentinel/dummy element.
inline uint32_t roundUp(uint32_t n, uint32_t m) { return ((n + m - 1) / m) * m; }
inline uint32_t capacityFor(uint32_t n) { return roundUp(n + 1, kThreadsPerGroup); }

inline uint32_t groupsFor(uint32_t threads) {
	return (threads + kThreadsPerGroup - 1) / kThreadsPerGroup;
}

#define VK_CHECK(expr)                                                       \
	do {                                                                     \
		VkResult _r = (expr);                                                \
		if (_r != VK_SUCCESS) {                                              \
			std::fprintf(stderr, "[avbd-vk] %s failed: %d\n", #expr, int(_r)); \
			return false;                                                    \
		}                                                                    \
	} while (0)

// Host-visible, host-coherent, persistently mapped buffer. This mirrors
// Metal's MTLResourceStorageModeShared + .contents: the Metal backend
// reads and writes live buffers from the CPU (coloring, setGammaScale,
// the stepBackward prologue/epilogue) with no map/unmap, and keeping
// that property lets those code paths port unchanged.
struct Buf {
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	void *mapped = nullptr;
	VkDeviceSize size = 0;

	bool valid() const { return buffer != VK_NULL_HANDLE; }
	float *f32() { return static_cast<float *>(mapped); }
	const float *f32() const { return static_cast<const float *>(mapped); }
	uint32_t *u32() { return static_cast<uint32_t *>(mapped); }
	const uint32_t *u32() const { return static_cast<const uint32_t *>(mapped); }
};

struct Kernel {
	VkShaderModule module = VK_NULL_HANDLE;
	VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	const avbd_table::KernelDesc *desc = nullptr;
};

// One name -> buffer override for a dispatch. The backward pass reuses
// the forward gather kernels against different buffers (e.g. binding
// the per-constraint position cotangents where springGradA normally
// goes), which is exactly what this expresses.
struct Override {
	const char *name;
	Buf *buf;
};

// Parameter names are NOT globally unique across kernels: `stiffness`
// means spring stiffness in spring_force and attachment stiffness in
// attachment_force_al, `idx` is triIdx in the membrane kernels and
// bendIdx in the bending ones, and so on. Binding by bare name would
// therefore silently bind the wrong buffer.
//
// The gather kernels already use disambiguated names (springGradA,
// vertTriOffset, ...) that match the registry directly; only the
// per-constraint force and dual-update kernels use generic ones, so the
// alias table below covers exactly those. An entry maps
// (kernel, shader param) -> canonical registry name.
struct Alias {
	const char *kernel;
	const char *param;
	const char *buffer;
};

const Alias kAliases[] = {
	// spring
	{"spring_force", "p1Idx", "springP1Idx"},
	{"spring_force", "p2Idx", "springP2Idx"},
	{"spring_force", "restLen", "springRestLen"},
	{"spring_force", "stiffness", "springStiffness"},
	{"spring_force", "gradA", "springGradA"},
	{"spring_force", "hess", "springHess"},
	{"spring_force_backward", "p1Idx", "springP1Idx"},
	{"spring_force_backward", "p2Idx", "springP2Idx"},
	{"spring_force_backward", "restLen", "springRestLen"},
	{"spring_force_backward", "stiffness", "springStiffness"},
	// attachment
	{"attachment_force_al", "vertIdx", "attachVertIdx"},
	{"attachment_force_al", "fixedPos", "attachFixedPos"},
	{"attachment_force_al", "stiffness", "attachStiffness"},
	{"attachment_force_al", "lambda", "attachLambda"},
	{"attachment_force_al", "gradV", "attachGradV"},
	{"attachment_force_al", "hessScalar", "attachHessScalar"},
	{"attachment_dual_update", "vertIdx", "attachVertIdx"},
	{"attachment_dual_update", "fixedPos", "attachFixedPos"},
	{"attachment_dual_update", "gamma", "attachGamma"},
	{"attachment_dual_update", "lambda", "attachLambda"},
	{"attachment_force_al_backward", "vertIdx", "attachVertIdx"},
	{"attachment_force_al_backward", "fixedPos", "attachFixedPos"},
	{"attachment_force_al_backward", "stiffness", "attachStiffness"},
	// triangle membrane
	{"triangle_membrane_force_al", "idx", "triIdx"},
	{"triangle_membrane_force_al", "stiffness", "triStiffness"},
	{"triangle_membrane_force_al", "lambda0", "triLambda0"},
	{"triangle_membrane_force_al", "lambda1", "triLambda1"},
	{"triangle_membrane_force_al", "grad", "triGrad"},
	{"triangle_membrane_force_al", "hessScalar", "triHessScalar"},
	{"triangle_membrane_force_al", "inv_deltaUV", "triInvUV"},
	{"triangle_membrane_dual_update", "idx", "triIdx"},
	{"triangle_membrane_dual_update", "gamma", "triGamma"},
	{"triangle_membrane_dual_update", "lambda0", "triLambda0"},
	{"triangle_membrane_dual_update", "lambda1", "triLambda1"},
	{"triangle_membrane_dual_update", "inv_deltaUV", "triInvUV"},
	{"triangle_membrane_force_al_backward", "idx", "triIdx"},
	{"triangle_membrane_force_al_backward", "stiffness", "triStiffness"},
	{"triangle_membrane_force_al_backward", "lambda0", "triLambda0"},
	{"triangle_membrane_force_al_backward", "lambda1", "triLambda1"},
	{"triangle_membrane_force_al_backward", "inv_deltaUV", "triInvUV"},
	// dihedral bending
	{"triangle_bending_force_al", "idx", "bendIdx"},
	{"triangle_bending_force_al", "weight", "bendWeight"},
	{"triangle_bending_force_al", "nTarget", "bendNTarget"},
	{"triangle_bending_force_al", "stiffness", "bendStiffness"},
	{"triangle_bending_force_al", "lambda", "bendLambda"},
	{"triangle_bending_force_al", "grad", "bendGrad"},
	{"triangle_bending_force_al", "hessScalar", "bendHessScalar"},
	{"triangle_bending_dual_update", "idx", "bendIdx"},
	{"triangle_bending_dual_update", "weight", "bendWeight"},
	{"triangle_bending_dual_update", "nTarget", "bendNTarget"},
	{"triangle_bending_dual_update", "gamma", "bendGamma"},
	{"triangle_bending_dual_update", "lambda", "bendLambda"},
	{"triangle_bending_force_al_backward", "idx", "bendIdx"},
	{"triangle_bending_force_al_backward", "weight", "bendWeight"},
	{"triangle_bending_force_al_backward", "nTarget", "bendNTarget"},
	{"triangle_bending_force_al_backward", "stiffness", "bendStiffness"},
	{"triangle_bending_force_al_backward", "lambda", "bendLambda"},
	// --- adjoint outputs. `v_stiffness`, `v_grad`, `v_hessScalar`,
	// `v_lambda` and `v_p` are each shared by several backward kernels
	// and must route to that constraint type's cotangent buffer.
	{"spring_force_backward", "v_gradA", "v_springGradA"},
	{"spring_force_backward", "v_p_d", "vSpringPd"},
	{"spring_force_backward", "v_restLen", "vSpringRestLen"},
	{"spring_force_backward", "v_stiffness", "vSpringStiff"},
	{"attachment_force_al_backward", "v_gradV", "v_attachGradV"},
	{"attachment_force_al_backward", "v_hessScalar", "v_attachHessScalar"},
	{"attachment_force_al_backward", "v_positions", "vPositionsGrad"},
	{"attachment_force_al_backward", "v_fixedPos", "vAttachFixedPos"},
	{"attachment_force_al_backward", "v_lambda", "vAttachLambda"},
	{"attachment_force_al_backward", "v_stiffness", "vAttachStiff"},
	{"triangle_membrane_force_al_backward", "v_grad", "v_triGrad"},
	{"triangle_membrane_force_al_backward", "v_hessScalar", "v_triHessScalar"},
	{"triangle_membrane_force_al_backward", "v_p", "vTriP"},
	{"triangle_membrane_force_al_backward", "v_stiffness", "vTriStiff"},
	{"triangle_membrane_force_al_backward", "v_lambda0", "vTriLambda0"},
	{"triangle_membrane_force_al_backward", "v_lambda1", "vTriLambda1"},
	{"triangle_bending_force_al_backward", "v_grad", "v_bendGrad"},
	{"triangle_bending_force_al_backward", "v_hessScalar", "v_bendHessScalar"},
	{"triangle_bending_force_al_backward", "v_p", "vBendP"},
	{"triangle_bending_force_al_backward", "v_nTarget", "vBendNTarget"},
	{"triangle_bending_force_al_backward", "v_stiffness", "vBendStiff"},
	{"triangle_bending_force_al_backward", "v_lambda", "vBendLambda"},
};

const char *aliasFor(const char *kernel, const std::string &param) {
	for (const Alias &a : kAliases) {
		if (param == a.param && std::strcmp(kernel, a.kernel) == 0) return a.buffer;
	}
	return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------

struct AvbdSolver::Impl {
	// --- Vulkan context ---
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t queueFamily = 0;
	VkCommandPool cmdPool = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties memProps{};
	VkDeviceSize uboAlignment = 256;

	std::vector<VkDescriptorPool> descPools;
	size_t descPoolCursor = 0;

	// Uniform ring for per-dispatch params blocks. Sub-allocated during
	// recording and reset per submission.
	Buf uboRing;
	VkDeviceSize uboCursor = 0;

	std::unordered_map<std::string, Kernel> kernels;
	std::vector<Buf *> ownedBuffers;

	// Named buffer registry: dispatch resolves each binding by the name
	// the shader declares, so binding order never has to be restated.
	std::unordered_map<std::string, Buf *> named;

	bool ok = false;
	bool meshReady = false;
	bool backwardReady = false;
	bool coloringBuilt = false;

	uint32_t nVerts = 0, nSprings = 0, nAttach = 0, nTri = 0, nBend = 0;
	float invHSqCached = 0.0f;
	uint32_t numColors = 1;
	// Offsets into the *padded* vertPerm; length numColors + 1, where
	// entry k+1 - k includes this color's tail padding.
	std::vector<uint32_t> colorStart;
	std::vector<uint32_t> colorCount;

	uint32_t selfCollK = 0;
	bool selfCollReady = false;

	// --- per-vertex ---
	Buf positions, predicted, mass, gScratch, hScratch, vertPerm, positionsPreStep;
	// --- spring ---
	Buf springP1Idx, springP2Idx, springRestLen, springStiffness;
	Buf springGradA, springHess;
	Buf vertSpringOffset, vertSpringIdx, vertSpringRole;
	// --- attachment ---
	Buf attachVertIdx, attachFixedPos, attachStiffness, attachGradV, attachHessScalar;
	Buf attachLambda, attachGamma;
	Buf vertAttachOffset, vertAttachIdx;
	// --- triangle ---
	Buf triIdx, triInvUV, triStiffness, triGrad, triHessScalar;
	Buf triLambda0, triLambda1, triGamma;
	Buf vertTriOffset, vertTriIdx, vertTriRole;
	// --- bending ---
	Buf bendIdx, bendWeight, bendNTarget, bendStiffness, bendGrad, bendHessScalar;
	Buf bendLambda, bendGamma;
	Buf vertBendOffset, vertBendIdx, vertBendRole;
	// --- self collision ---
	Buf radii, neighbors;
	// --- backward ---
	Buf vPositionsLoss, vG, vH, deltaX, vPositionsGrad, vPositionsInit;
	Buf vPredicted, vMass, hScratchJunk;
	Buf vSpringGradA, vSpringHess, vSpringPd, vSpringRestLen, vSpringStiff;
	Buf vAttachGradV, vAttachHessScalar, vAttachFixedPos, vAttachStiff, vAttachLambda;
	Buf vTriGrad, vTriHessScalar, vTriP, vTriStiff, vTriLambda0, vTriLambda1;
	Buf vBendGrad, vBendHessScalar, vBendP, vBendNTarget, vBendStiff, vBendLambda;

	// ---- lifecycle ----
	bool initVulkan();
	bool loadKernels(const std::string &dir);
	void destroy();

	// ---- buffers ----
	uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags want) const;
	bool createBuffer(Buf &b, VkDeviceSize bytes, VkBufferUsageFlags usage);
	void destroyBuffer(Buf &b);
	// Allocates (or reallocates) `b` to `bytes`, zero-filled, and registers
	// it under `name` so dispatch() can resolve it.
	bool alloc(Buf &b, const char *name, VkDeviceSize bytes);

	// ---- dispatch ----
	VkCommandBuffer begin();
	bool endAndWait(VkCommandBuffer cmd);
	void barrier(VkCommandBuffer cmd);
	VkDescriptorSet allocDescriptorSet(VkDescriptorSetLayout layout);
	bool dispatch(VkCommandBuffer cmd, const char *kernelName, uint32_t threads,
			const void *params = nullptr, size_t paramsSize = 0,
			std::initializer_list<Override> overrides = {});

	Buf *resolve(const char *kernel, const std::string &param,
			std::initializer_list<Override> overrides) const;

	void buildVertexColoringIfNeeded();
	bool ensureBackwardBuffers();
};

// ---------------------------------------------------------------------
// Vulkan setup
// ---------------------------------------------------------------------

bool AvbdSolver::Impl::initVulkan() {
	VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "cloth-avbd";
	// slangc emits SPIR-V 1.4, which Vulkan 1.1 does not accept (it caps
	// at SPIR-V 1.3). Requesting 1.1 happens to work on permissive
	// drivers but spirv-val rejects every module; 1.2 is the real
	// minimum. All 24 AVBD modules validate clean at 1.2 and 1.3.
	app.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;

	// AVBD_VK_VALIDATION=1 turns on the Khronos validation layer. Worth
	// it whenever touching the padding or barrier logic: both failure
	// modes are silent numerically but loud to the validator.
	const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
	if (std::getenv("AVBD_VK_VALIDATION")) {
		ici.enabledLayerCount = 1;
		ici.ppEnabledLayerNames = layers;
	}
	VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

	uint32_t count = 0;
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
	if (count == 0) {
		std::fprintf(stderr, "[avbd-vk] no Vulkan physical devices\n");
		return false;
	}
	std::vector<VkPhysicalDevice> devices(count);
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

	// Prefer a discrete GPU; AVBD_VK_DEVICE=<index> overrides.
	int forced = -1;
	if (const char *e = std::getenv("AVBD_VK_DEVICE")) forced = std::atoi(e);
	if (forced >= 0 && forced < int(count)) {
		physical = devices[size_t(forced)];
	} else {
		physical = devices[0];
		for (VkPhysicalDevice d : devices) {
			VkPhysicalDeviceProperties p{};
			vkGetPhysicalDeviceProperties(d, &p);
			if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				physical = d;
				break;
			}
		}
	}

	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(physical, &props);
	uboAlignment = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
	vkGetPhysicalDeviceMemoryProperties(physical, &memProps);

	uint32_t qCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &qCount, nullptr);
	std::vector<VkQueueFamilyProperties> qs(qCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &qCount, qs.data());
	bool found = false;
	for (uint32_t i = 0; i < qCount; ++i) {
		if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			queueFamily = i;
			found = true;
			break;
		}
	}
	if (!found) {
		std::fprintf(stderr, "[avbd-vk] no compute queue family\n");
		return false;
	}

	const float prio = 1.0f;
	VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = queueFamily;
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;

	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	VK_CHECK(vkCreateDevice(physical, &dci, nullptr, &device));
	vkGetDeviceQueue(device, queueFamily, 0, &queue);

	VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	cpci.queueFamilyIndex = queueFamily;
	cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cmdPool));

	if (!createBuffer(uboRing, 1u << 20, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;

	std::fprintf(stderr, "[avbd-vk] device: %s (Vulkan %u.%u)\n", props.deviceName,
			VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion));
	return true;
}

uint32_t AvbdSolver::Impl::findMemoryType(uint32_t bits, VkMemoryPropertyFlags want) const {
	for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
		if ((bits & (1u << i)) &&
				(memProps.memoryTypes[i].propertyFlags & want) == want) {
			return i;
		}
	}
	return UINT32_MAX;
}

bool AvbdSolver::Impl::createBuffer(Buf &b, VkDeviceSize bytes, VkBufferUsageFlags usage) {
	destroyBuffer(b);
	if (bytes == 0) bytes = 4;

	VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.size = bytes;
	bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &b.buffer));

	VkMemoryRequirements req{};
	vkGetBufferMemoryRequirements(device, b.buffer, &req);

	// Prefer DEVICE_LOCAL|HOST_VISIBLE (ReBAR) so the CPU-side buffer
	// access the Metal backend relies on stays cheap; fall back to plain
	// host-visible if the device does not expose a ReBAR heap.
	uint32_t type = findMemoryType(req.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
					VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	if (type == UINT32_MAX) {
		type = findMemoryType(req.memoryTypeBits,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	}
	if (type == UINT32_MAX) {
		std::fprintf(stderr, "[avbd-vk] no host-visible memory type\n");
		return false;
	}

	VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = type;
	VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &b.memory));
	VK_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
	VK_CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
	b.size = bytes;
	std::memset(b.mapped, 0, size_t(bytes));
	return true;
}

void AvbdSolver::Impl::destroyBuffer(Buf &b) {
	if (b.mapped) {
		vkUnmapMemory(device, b.memory);
		b.mapped = nullptr;
	}
	if (b.buffer) {
		vkDestroyBuffer(device, b.buffer, nullptr);
		b.buffer = VK_NULL_HANDLE;
	}
	if (b.memory) {
		vkFreeMemory(device, b.memory, nullptr);
		b.memory = VK_NULL_HANDLE;
	}
	b.size = 0;
}

bool AvbdSolver::Impl::alloc(Buf &b, const char *name, VkDeviceSize bytes) {
	if (!createBuffer(b, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
	named[name] = &b;
	if (std::find(ownedBuffers.begin(), ownedBuffers.end(), &b) == ownedBuffers.end()) {
		ownedBuffers.push_back(&b);
	}
	return true;
}

// ---------------------------------------------------------------------
// Kernel loading
// ---------------------------------------------------------------------

bool AvbdSolver::Impl::loadKernels(const std::string &dir) {
	for (uint32_t i = 0; i < avbd_table::kNumKernels; ++i) {
		const avbd_table::KernelDesc &kd = avbd_table::kKernels[i];
		const std::string path = dir + "/" + kd.name + ".spv";

		std::FILE *f = std::fopen(path.c_str(), "rb");
		if (!f) {
			std::fprintf(stderr, "[avbd-vk] missing SPIR-V module: %s\n", path.c_str());
			return false;
		}
		std::fseek(f, 0, SEEK_END);
		const long len = std::ftell(f);
		std::fseek(f, 0, SEEK_SET);
		if (len <= 0 || (len % 4) != 0) {
			std::fprintf(stderr, "[avbd-vk] bad SPIR-V size for %s: %ld\n", kd.name, len);
			std::fclose(f);
			return false;
		}
		std::vector<uint32_t> code(size_t(len) / 4);
		const size_t got = std::fread(code.data(), 1, size_t(len), f);
		std::fclose(f);
		if (got != size_t(len)) {
			std::fprintf(stderr, "[avbd-vk] short read on %s\n", path.c_str());
			return false;
		}

		Kernel k;
		k.desc = &kd;

		VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		smci.codeSize = size_t(len);
		smci.pCode = code.data();
		VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &k.module));

		std::vector<VkDescriptorSetLayoutBinding> binds(kd.nBindings);
		for (uint32_t b = 0; b < kd.nBindings; ++b) {
			binds[b] = {};
			binds[b].binding = kd.bindings[b].binding;
			binds[b].descriptorType = kd.bindings[b].type;
			binds[b].descriptorCount = 1;
			binds[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		VkDescriptorSetLayoutCreateInfo dslci{
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		dslci.bindingCount = uint32_t(binds.size());
		dslci.pBindings = binds.data();
		VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &k.setLayout));

		VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		plci.setLayoutCount = 1;
		plci.pSetLayouts = &k.setLayout;
		VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &k.pipelineLayout));

		VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		cpci.stage.module = k.module;
		cpci.stage.pName = kd.entry;
		cpci.layout = k.pipelineLayout;
		VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr,
				&k.pipeline));

		kernels[kd.name] = k;
	}
	return true;
}

void AvbdSolver::Impl::destroy() {
	if (device == VK_NULL_HANDLE) {
		if (instance) vkDestroyInstance(instance, nullptr);
		return;
	}
	vkDeviceWaitIdle(device);

	for (auto &kv : kernels) {
		Kernel &k = kv.second;
		if (k.pipeline) vkDestroyPipeline(device, k.pipeline, nullptr);
		if (k.pipelineLayout) vkDestroyPipelineLayout(device, k.pipelineLayout, nullptr);
		if (k.setLayout) vkDestroyDescriptorSetLayout(device, k.setLayout, nullptr);
		if (k.module) vkDestroyShaderModule(device, k.module, nullptr);
	}
	kernels.clear();

	for (Buf *b : ownedBuffers) destroyBuffer(*b);
	ownedBuffers.clear();
	destroyBuffer(uboRing);

	for (VkDescriptorPool p : descPools) vkDestroyDescriptorPool(device, p, nullptr);
	descPools.clear();

	if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
	vkDestroyDevice(device, nullptr);
	if (instance) vkDestroyInstance(instance, nullptr);
	device = VK_NULL_HANDLE;
	instance = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------

VkCommandBuffer AvbdSolver::Impl::begin() {
	// Reset the transient allocators that live for one submission.
	uboCursor = 0;
	descPoolCursor = 0;
	for (VkDescriptorPool p : descPools) vkResetDescriptorPool(device, p, 0);

	VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cbai.commandPool = cmdPool;
	cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbai.commandBufferCount = 1;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) return VK_NULL_HANDLE;

	VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return VK_NULL_HANDLE;
	return cmd;
}

bool AvbdSolver::Impl::endAndWait(VkCommandBuffer cmd) {
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence = VK_NULL_HANDLE;
	VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));

	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	VkResult r = vkQueueSubmit(queue, 1, &si, fence);
	if (r == VK_SUCCESS) {
		r = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
	}
	vkDestroyFence(device, fence, nullptr);
	vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
	if (r != VK_SUCCESS) {
		std::fprintf(stderr, "[avbd-vk] submit/wait failed: %d\n", int(r));
		return false;
	}
	return true;
}

void AvbdSolver::Impl::barrier(VkCommandBuffer cmd) {
	// Unconditional full shader-write -> shader-read barrier between
	// every pair of dispatches. The Metal backend gets this for free
	// from MTLDispatchTypeSerial; here it must be explicit, and the
	// dependency graph (init -> gathers, force -> its gather, gathers ->
	// solve_apply, solve_apply -> next color's force kernels) touches
	// essentially every adjacent pair anyway.
	VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

VkDescriptorSet AvbdSolver::Impl::allocDescriptorSet(VkDescriptorSetLayout layout) {
	for (;;) {
		if (descPoolCursor >= descPools.size()) {
			VkDescriptorPoolSize sizes[2] = {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
				{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512},
			};
			VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
			dpci.maxSets = 512;
			dpci.poolSizeCount = 2;
			dpci.pPoolSizes = sizes;
			VkDescriptorPool pool = VK_NULL_HANDLE;
			if (vkCreateDescriptorPool(device, &dpci, nullptr, &pool) != VK_SUCCESS) {
				return VK_NULL_HANDLE;
			}
			descPools.push_back(pool);
		}
		VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		dsai.descriptorPool = descPools[descPoolCursor];
		dsai.descriptorSetCount = 1;
		dsai.pSetLayouts = &layout;
		VkDescriptorSet set = VK_NULL_HANDLE;
		const VkResult r = vkAllocateDescriptorSets(device, &dsai, &set);
		if (r == VK_SUCCESS) return set;
		if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) {
			return VK_NULL_HANDLE;
		}
		// This pool is full -- move to (or create) the next one.
		++descPoolCursor;
	}
}

Buf *AvbdSolver::Impl::resolve(const char *kernel, const std::string &param,
		std::initializer_list<Override> overrides) const {
	// Overrides are keyed on the shader's own parameter name, so a call
	// site can redirect one binding without knowing the registry name.
	for (const Override &o : overrides) {
		if (param == o.name) return o.buf;
	}
	const char *canonical = aliasFor(kernel, param);
	auto it = named.find(canonical ? std::string(canonical) : param);
	return it == named.end() ? nullptr : it->second;
}

bool AvbdSolver::Impl::dispatch(VkCommandBuffer cmd, const char *kernelName,
		uint32_t threads, const void *params, size_t paramsSize,
		std::initializer_list<Override> overrides) {
	if (threads == 0) return true;

	auto it = kernels.find(kernelName);
	if (it == kernels.end()) {
		std::fprintf(stderr, "[avbd-vk] unknown kernel %s\n", kernelName);
		return false;
	}
	const Kernel &k = it->second;
	const avbd_table::KernelDesc &kd = *k.desc;

	VkDescriptorSet set = allocDescriptorSet(k.setLayout);
	if (set == VK_NULL_HANDLE) {
		std::fprintf(stderr, "[avbd-vk] descriptor alloc failed for %s\n", kernelName);
		return false;
	}

	std::vector<VkDescriptorBufferInfo> infos(kd.nBindings);
	std::vector<VkWriteDescriptorSet> writes(kd.nBindings);

	for (uint32_t b = 0; b < kd.nBindings; ++b) {
		const avbd_table::BindingDesc &bd = kd.bindings[b];
		infos[b] = {};

		if (int32_t(bd.binding) == kd.paramsBinding) {
			if (!params || paramsSize == 0) {
				std::fprintf(stderr, "[avbd-vk] %s needs a params block\n", kernelName);
				return false;
			}
			// Sub-allocate the params block out of the uniform ring.
			const VkDeviceSize off =
					(uboCursor + uboAlignment - 1) / uboAlignment * uboAlignment;
			if (off + paramsSize > uboRing.size) {
				std::fprintf(stderr, "[avbd-vk] uniform ring exhausted\n");
				return false;
			}
			std::memcpy(static_cast<char *>(uboRing.mapped) + off, params, paramsSize);
			uboCursor = off + paramsSize;
			infos[b].buffer = uboRing.buffer;
			infos[b].offset = off;
			infos[b].range = paramsSize;
		} else {
			Buf *buf = resolve(kernelName, bd.name, overrides);
			if (!buf || !buf->valid()) {
				std::fprintf(stderr, "[avbd-vk] %s: unbound buffer '%s'\n", kernelName,
						bd.name);
				return false;
			}
			infos[b].buffer = buf->buffer;
			infos[b].offset = 0;
			infos[b].range = VK_WHOLE_SIZE;
		}

		writes[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		writes[b].dstSet = set;
		writes[b].dstBinding = bd.binding;
		writes[b].descriptorCount = 1;
		writes[b].descriptorType = bd.type;
		writes[b].pBufferInfo = &infos[b];
	}
	vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1,
			&set, 0, nullptr);
	vkCmdDispatch(cmd, groupsFor(threads), 1, 1);
	barrier(cmd);
	return true;
}

// ---------------------------------------------------------------------
// Params blocks. Layouts match the reflected ConstantBuffer fields, and
// are identical to the Metal backend's structs.
// ---------------------------------------------------------------------

namespace {

struct VbdInitParams {
	float invHSquared;
	uint32_t colorOffset;
	// MUST match the kernel's params struct exactly. The kernels gained
	// a `count` field with the tid bounds guard; omitting it here left
	// them reading the guard bound past the end of an undersized
	// uniform buffer.
	uint32_t count;
};
struct VbdGatherParams {
	uint32_t colorOffset;
	uint32_t count;
};
struct VbdSolveApplyParams {
	uint32_t colorOffset;
	// Bounds for the kernel's own tid guard. vkCmdDispatch rounds up to
	// whole workgroups, so the kernel needs to know where the colour's
	// range actually ends.
	uint32_t count;
};
struct SelfCollisionScanParams {
	uint32_t nVerts;
	uint32_t K;
};
struct VbdInitBackwardParams {
	float invHSquared;
};

constexpr uint32_t kSentinel = 0xFFFFFFFFu;

// float3 in a StructuredBuffer has a 16-byte stride, so host-side tight
// xyz triples are expanded on upload and compacted on readback. Same
// convention as the Metal backend's uploadVec3Padded / readVec3Padded.
void uploadVec3Padded(Buf &b, const float *src, uint32_t n) {
	if (!b.valid() || !src) return;
	float *dst = b.f32();
	for (uint32_t i = 0; i < n; ++i) {
		dst[4 * i + 0] = src[3 * i + 0];
		dst[4 * i + 1] = src[3 * i + 1];
		dst[4 * i + 2] = src[3 * i + 2];
		dst[4 * i + 3] = 0.0f;
	}
}

void readVec3Padded(const Buf &b, std::vector<float> &out, uint32_t n) {
	out.assign(size_t(3) * n, 0.0f);
	if (!b.valid()) return;
	const float *src = b.f32();
	for (uint32_t i = 0; i < n; ++i) {
		out[3 * i + 0] = src[4 * i + 0];
		out[3 * i + 1] = src[4 * i + 1];
		out[3 * i + 2] = src[4 * i + 2];
	}
}

void readScalars(const Buf &b, std::vector<float> &out, uint32_t n) {
	out.assign(n, 0.0f);
	if (b.valid() && n) std::memcpy(out.data(), b.mapped, size_t(n) * sizeof(float));
}

}  // namespace

// ---------------------------------------------------------------------
// Coloring
// ---------------------------------------------------------------------

void AvbdSolver::Impl::buildVertexColoringIfNeeded() {
	if (coloringBuilt || nVerts == 0) return;

	// Adjacency from every constraint's clique: springs are 2-cliques,
	// triangles 3-cliques, bending stencils 4-cliques. Attachments touch
	// one vertex each and so constrain nothing.
	std::vector<std::vector<uint32_t>> adj(nVerts);
	auto link = [&](uint32_t a, uint32_t b) {
		if (a == b || a >= nVerts || b >= nVerts) return;
		adj[a].push_back(b);
		adj[b].push_back(a);
	};
	if (nSprings) {
		const uint32_t *p1 = springP1Idx.u32();
		const uint32_t *p2 = springP2Idx.u32();
		for (uint32_t i = 0; i < nSprings; ++i) link(p1[i], p2[i]);
	}
	if (nTri) {
		const uint32_t *t = triIdx.u32();
		for (uint32_t i = 0; i < nTri; ++i) {
			link(t[3 * i + 0], t[3 * i + 1]);
			link(t[3 * i + 1], t[3 * i + 2]);
			link(t[3 * i + 2], t[3 * i + 0]);
		}
	}
	if (nBend) {
		const uint32_t *b = bendIdx.u32();
		for (uint32_t i = 0; i < nBend; ++i) {
			for (uint32_t r = 0; r < 4; ++r) {
				for (uint32_t s = r + 1; s < 4; ++s) link(b[4 * i + r], b[4 * i + s]);
			}
		}
	}

	// Greedy first-fit.
	std::vector<uint32_t> color(nVerts, kSentinel);
	std::vector<char> used;
	uint32_t nc = 0;
	for (uint32_t v = 0; v < nVerts; ++v) {
		used.assign(nc + 1, 0);
		for (uint32_t n : adj[v]) {
			if (color[n] != kSentinel && color[n] < used.size()) used[color[n]] = 1;
		}
		uint32_t c = 0;
		while (c < used.size() && used[c]) ++c;
		color[v] = c;
		if (c + 1 > nc) nc = c + 1;
	}

	std::vector<uint32_t> counts(nc, 0u);
	for (uint32_t v = 0; v < nVerts; ++v) counts[color[v]]++;

	// Lay each color out on a 64-lane boundary and pad its tail with the
	// sentinel vertex, so the ragged final group of every dispatch reads
	// and writes only the dummy slot instead of spilling into the next
	// color's vertices.
	colorStart.assign(nc, 0u);
	colorCount.assign(nc, 0u);
	uint32_t cursor = 0;
	for (uint32_t c = 0; c < nc; ++c) {
		colorStart[c] = cursor;
		colorCount[c] = counts[c];
		cursor += roundUp(counts[c], kThreadsPerGroup);
	}
	const uint32_t permLen = std::max(cursor, kThreadsPerGroup);

	// AVBD_VK_NO_PADDING=1 fills the colour tail with vertex 0 instead
	// of the sentinel. That is the exact corruption the padding exists
	// to prevent -- an overrun lane then addresses a REAL vertex. It is
	// a falsifiability switch: with the kernels' own `lane >= count`
	// guard in place the tail lanes return before touching vertPerm, so
	// results must be unchanged; without the guard this reproduces the
	// 324-component drift the colored conformance test reports.
	const bool noPad = (std::getenv("AVBD_VK_NO_PADDING") != nullptr);
	std::vector<uint32_t> perm(permLen, noPad ? 0u : nVerts);
	std::vector<uint32_t> fill(nc, 0u);
	for (uint32_t v = 0; v < nVerts; ++v) {
		const uint32_t c = color[v];
		perm[colorStart[c] + fill[c]++] = v;
	}

	if (!alloc(vertPerm, "vertPerm", VkDeviceSize(permLen) * sizeof(uint32_t))) return;
	std::memcpy(vertPerm.mapped, perm.data(), permLen * sizeof(uint32_t));

	numColors = nc;
	coloringBuilt = true;
	std::fprintf(stderr, "[avbd-vk] coloring: %u colors over %u verts\n", nc, nVerts);
}

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

AvbdSolver::AvbdSolver(const char *spirvPath) : impl_(new Impl()) {
	if (!impl_->initVulkan()) return;
	if (!impl_->loadKernels(spirvPath ? spirvPath : ".")) return;
	impl_->ok = true;
}

AvbdSolver::~AvbdSolver() {
	if (impl_) {
		impl_->destroy();
		delete impl_;
		impl_ = nullptr;
	}
}

bool AvbdSolver::ok() const { return impl_ && impl_->ok; }

void AvbdSolver::setupMesh(uint32_t nVerts, const float *positions, const float *predicted,
		const float *mass, float invHSquared) {
	if (!ok()) return;
	Impl &d = *impl_;
	d.nVerts = nVerts;
	d.invHSqCached = invHSquared;

	const uint32_t vcap = capacityFor(nVerts);
	const VkDeviceSize vec3Bytes = VkDeviceSize(vcap) * 4 * sizeof(float);
	const VkDeviceSize scalarBytes = VkDeviceSize(vcap) * sizeof(float);
	const VkDeviceSize hessBytes = VkDeviceSize(vcap) * 6 * sizeof(float);

	if (!d.alloc(d.positions, "positions", vec3Bytes)) return;
	if (!d.alloc(d.predicted, "predicted", vec3Bytes)) return;
	if (!d.alloc(d.gScratch, "gScratch", vec3Bytes)) return;
	if (!d.alloc(d.positionsPreStep, "positionsPreStep", vec3Bytes)) return;
	if (!d.alloc(d.mass, "mass", scalarBytes)) return;
	if (!d.alloc(d.hScratch, "hScratch", hessBytes)) return;

	uploadVec3Padded(d.positions, positions, nVerts);
	uploadVec3Padded(d.predicted, predicted, nVerts);
	if (mass) std::memcpy(d.mass.mapped, mass, size_t(nVerts) * sizeof(float));

	// Default: one color spanning every vertex with an identity
	// permutation, padded to a whole number of workgroups. Equivalent to
	// the pre-coloring block-Jacobi sweep. buildColoring() replaces this.
	const uint32_t permLen = roundUp(nVerts, kThreadsPerGroup);
	if (!d.alloc(d.vertPerm, "vertPerm",
				VkDeviceSize(std::max(permLen, kThreadsPerGroup)) * sizeof(uint32_t))) {
		return;
	}
	uint32_t *perm = d.vertPerm.u32();
	for (uint32_t i = 0; i < nVerts; ++i) perm[i] = i;
	for (uint32_t i = nVerts; i < permLen; ++i) perm[i] = nVerts;  // sentinel

	d.numColors = 1;
	d.colorStart.assign(1, 0u);
	d.colorCount.assign(1, nVerts);
	d.coloringBuilt = false;
	d.meshReady = true;
}

void AvbdSolver::uploadSprings(uint32_t nSprings, const uint32_t *p1Idx,
		const uint32_t *p2Idx, const float *restLen, const float *stiffness) {
	if (!ok()) return;
	Impl &d = *impl_;
	d.nSprings = nSprings;

	const uint32_t ccap = capacityFor(nSprings);
	if (!d.alloc(d.springP1Idx, "springP1Idx", VkDeviceSize(ccap) * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.springP2Idx, "springP2Idx", VkDeviceSize(ccap) * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.springRestLen, "springRestLen", VkDeviceSize(ccap) * sizeof(float)))
		return;
	if (!d.alloc(d.springStiffness, "springStiffness", VkDeviceSize(ccap) * sizeof(float)))
		return;
	if (!d.alloc(d.springGradA, "springGradA", VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.springHess, "springHess", VkDeviceSize(ccap) * 6 * sizeof(float)))
		return;

	std::memcpy(d.springP1Idx.mapped, p1Idx, size_t(nSprings) * sizeof(uint32_t));
	std::memcpy(d.springP2Idx.mapped, p2Idx, size_t(nSprings) * sizeof(uint32_t));
	std::memcpy(d.springRestLen.mapped, restLen, size_t(nSprings) * sizeof(float));
	std::memcpy(d.springStiffness.mapped, stiffness, size_t(nSprings) * sizeof(float));
	// Padded springs point at the dummy vertex with zero stiffness, so
	// tail lanes compute a zero force into a slot nothing reads.
	uint32_t *p1 = d.springP1Idx.u32();
	uint32_t *p2 = d.springP2Idx.u32();
	for (uint32_t i = nSprings; i < ccap; ++i) {
		p1[i] = d.nVerts;
		p2[i] = d.nVerts;
	}

	std::vector<uint32_t> pairs(size_t(nSprings) * 2);
	for (uint32_t i = 0; i < nSprings; ++i) {
		pairs[2 * i + 0] = p1Idx[i];
		pairs[2 * i + 1] = p2Idx[i];
	}
	std::vector<uint32_t> offsets, idx, role;
	buildCsr(d.nVerts, capacityFor(d.nVerts), pairs.data(), nSprings, 2, offsets, idx, role);

	if (!d.alloc(d.vertSpringOffset, "vertSpringOffset",
				offsets.size() * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.vertSpringIdx, "vertSpringIdx",
				std::max<size_t>(idx.size(), 1) * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.vertSpringRole, "vertSpringRole",
				std::max<size_t>(role.size(), 1) * sizeof(uint32_t)))
		return;
	std::memcpy(d.vertSpringOffset.mapped, offsets.data(), offsets.size() * sizeof(uint32_t));
	if (!idx.empty()) {
		std::memcpy(d.vertSpringIdx.mapped, idx.data(), idx.size() * sizeof(uint32_t));
		std::memcpy(d.vertSpringRole.mapped, role.data(), role.size() * sizeof(uint32_t));
	}
	d.coloringBuilt = false;
}

void AvbdSolver::uploadAttachments(uint32_t nAttach, const uint32_t *vertIdx,
		const float *fixedPos, const float *stiffness) {
	if (!ok()) return;
	Impl &d = *impl_;
	d.nAttach = nAttach;

	const uint32_t ccap = capacityFor(nAttach);
	if (!d.alloc(d.attachVertIdx, "attachVertIdx", VkDeviceSize(ccap) * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.attachFixedPos, "attachFixedPos",
				VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.attachStiffness, "attachStiffness", VkDeviceSize(ccap) * sizeof(float)))
		return;
	if (!d.alloc(d.attachGradV, "attachGradV", VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.attachHessScalar, "attachHessScalar",
				VkDeviceSize(ccap) * sizeof(float)))
		return;
	if (!d.alloc(d.attachLambda, "attachLambda", VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.attachGamma, "attachGamma", VkDeviceSize(ccap) * sizeof(float))) return;

	std::memcpy(d.attachVertIdx.mapped, vertIdx, size_t(nAttach) * sizeof(uint32_t));
	uploadVec3Padded(d.attachFixedPos, fixedPos, nAttach);
	std::memcpy(d.attachStiffness.mapped, stiffness, size_t(nAttach) * sizeof(float));
	// gamma starts at the constraint stiffness (see setGammaScale).
	std::memcpy(d.attachGamma.mapped, stiffness, size_t(nAttach) * sizeof(float));
	uint32_t *vi = d.attachVertIdx.u32();
	for (uint32_t i = nAttach; i < ccap; ++i) vi[i] = d.nVerts;

	std::vector<uint32_t> offsets, idx, role;
	buildCsr(d.nVerts, capacityFor(d.nVerts), vertIdx, nAttach, 1, offsets, idx, role);
	if (!d.alloc(d.vertAttachOffset, "vertAttachOffset",
				offsets.size() * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.vertAttachIdx, "vertAttachIdx",
				std::max<size_t>(idx.size(), 1) * sizeof(uint32_t)))
		return;
	std::memcpy(d.vertAttachOffset.mapped, offsets.data(), offsets.size() * sizeof(uint32_t));
	if (!idx.empty()) {
		std::memcpy(d.vertAttachIdx.mapped, idx.data(), idx.size() * sizeof(uint32_t));
	}
}

void AvbdSolver::updateAttachmentFixedPos(const float *fixedPos) {
	if (!ok() || impl_->nAttach == 0 || !impl_->attachFixedPos.valid()) return;
	uploadVec3Padded(impl_->attachFixedPos, fixedPos, impl_->nAttach);
}

void AvbdSolver::uploadTriangles(uint32_t nTri, const uint32_t *triIdx, const float *invUV,
		const float *stiffness) {
	if (!ok()) return;
	Impl &d = *impl_;
	d.nTri = nTri;

	const uint32_t ccap = capacityFor(nTri);
	if (!d.alloc(d.triIdx, "triIdx", VkDeviceSize(ccap) * 3 * sizeof(uint32_t))) return;
	if (!d.alloc(d.triInvUV, "triInvUV", VkDeviceSize(ccap) * 4 * sizeof(float))) return;
	if (!d.alloc(d.triStiffness, "triStiffness", VkDeviceSize(ccap) * sizeof(float)))
		return;
	if (!d.alloc(d.triGrad, "triGrad", VkDeviceSize(ccap) * 3 * 4 * sizeof(float))) return;
	if (!d.alloc(d.triHessScalar, "triHessScalar", VkDeviceSize(ccap) * 3 * sizeof(float)))
		return;
	if (!d.alloc(d.triLambda0, "triLambda0", VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.triLambda1, "triLambda1", VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.triGamma, "triGamma", VkDeviceSize(ccap) * sizeof(float))) return;

	std::memcpy(d.triIdx.mapped, triIdx, size_t(nTri) * 3 * sizeof(uint32_t));
	std::memcpy(d.triInvUV.mapped, invUV, size_t(nTri) * 4 * sizeof(float));
	std::memcpy(d.triStiffness.mapped, stiffness, size_t(nTri) * sizeof(float));
	std::memcpy(d.triGamma.mapped, stiffness, size_t(nTri) * sizeof(float));
	uint32_t *ti = d.triIdx.u32();
	for (uint32_t i = nTri; i < ccap; ++i) {
		ti[3 * i + 0] = d.nVerts;
		ti[3 * i + 1] = d.nVerts;
		ti[3 * i + 2] = d.nVerts;
	}

	std::vector<uint32_t> offsets, idx, role;
	buildCsr(d.nVerts, capacityFor(d.nVerts), triIdx, nTri, 3, offsets, idx, role);
	if (!d.alloc(d.vertTriOffset, "vertTriOffset", offsets.size() * sizeof(uint32_t))) return;
	if (!d.alloc(d.vertTriIdx, "vertTriIdx",
				std::max<size_t>(idx.size(), 1) * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.vertTriRole, "vertTriRole",
				std::max<size_t>(role.size(), 1) * sizeof(uint32_t)))
		return;
	std::memcpy(d.vertTriOffset.mapped, offsets.data(), offsets.size() * sizeof(uint32_t));
	if (!idx.empty()) {
		std::memcpy(d.vertTriIdx.mapped, idx.data(), idx.size() * sizeof(uint32_t));
		std::memcpy(d.vertTriRole.mapped, role.data(), role.size() * sizeof(uint32_t));
	}
	d.coloringBuilt = false;
}

void AvbdSolver::uploadBendings(uint32_t nBend, const uint32_t *bendIdx, const float *weight,
		const float *nTarget, const float *stiffness) {
	if (!ok()) return;
	Impl &d = *impl_;
	d.nBend = nBend;

	const uint32_t ccap = capacityFor(nBend);
	if (!d.alloc(d.bendIdx, "bendIdx", VkDeviceSize(ccap) * 4 * sizeof(uint32_t))) return;
	if (!d.alloc(d.bendWeight, "bendWeight", VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.bendNTarget, "bendNTarget", VkDeviceSize(ccap) * sizeof(float))) return;
	if (!d.alloc(d.bendStiffness, "bendStiffness", VkDeviceSize(ccap) * sizeof(float)))
		return;
	if (!d.alloc(d.bendGrad, "bendGrad", VkDeviceSize(ccap) * 4 * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.bendHessScalar, "bendHessScalar",
				VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.bendLambda, "bendLambda", VkDeviceSize(ccap) * 4 * sizeof(float)))
		return;
	if (!d.alloc(d.bendGamma, "bendGamma", VkDeviceSize(ccap) * sizeof(float))) return;

	std::memcpy(d.bendIdx.mapped, bendIdx, size_t(nBend) * 4 * sizeof(uint32_t));
	std::memcpy(d.bendWeight.mapped, weight, size_t(nBend) * 4 * sizeof(float));
	std::memcpy(d.bendNTarget.mapped, nTarget, size_t(nBend) * sizeof(float));
	std::memcpy(d.bendStiffness.mapped, stiffness, size_t(nBend) * sizeof(float));
	std::memcpy(d.bendGamma.mapped, stiffness, size_t(nBend) * sizeof(float));
	uint32_t *bi = d.bendIdx.u32();
	for (uint32_t i = nBend; i < ccap; ++i) {
		for (uint32_t r = 0; r < 4; ++r) bi[4 * i + r] = d.nVerts;
	}

	std::vector<uint32_t> offsets, idx, role;
	buildCsr(d.nVerts, capacityFor(d.nVerts), bendIdx, nBend, 4, offsets, idx, role);
	if (!d.alloc(d.vertBendOffset, "vertBendOffset", offsets.size() * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.vertBendIdx, "vertBendIdx",
				std::max<size_t>(idx.size(), 1) * sizeof(uint32_t)))
		return;
	if (!d.alloc(d.vertBendRole, "vertBendRole",
				std::max<size_t>(role.size(), 1) * sizeof(uint32_t)))
		return;
	std::memcpy(d.vertBendOffset.mapped, offsets.data(), offsets.size() * sizeof(uint32_t));
	if (!idx.empty()) {
		std::memcpy(d.vertBendIdx.mapped, idx.data(), idx.size() * sizeof(uint32_t));
		std::memcpy(d.vertBendRole.mapped, role.data(), role.size() * sizeof(uint32_t));
	}
	d.coloringBuilt = false;
}

void AvbdSolver::buildColoring() {
	if (ok()) impl_->buildVertexColoringIfNeeded();
}

void AvbdSolver::setGammaScale(float scale) {
	if (!ok()) return;
	Impl &d = *impl_;
	auto scaleBuf = [&](Buf &b, uint32_t n) {
		if (!b.valid()) return;
		float *g = b.f32();
		for (uint32_t i = 0; i < n; ++i) g[i] *= scale;
	};
	scaleBuf(d.attachGamma, d.nAttach);
	scaleBuf(d.triGamma, d.nTri);
	scaleBuf(d.bendGamma, d.nBend);
}

void AvbdSolver::updateState(const float *positions, const float *predicted) {
	if (!ok() || !impl_->meshReady) return;
	uploadVec3Padded(impl_->positions, positions, impl_->nVerts);
	uploadVec3Padded(impl_->predicted, predicted, impl_->nVerts);
}

// ---------------------------------------------------------------------
// Forward step
// ---------------------------------------------------------------------

int AvbdSolver::step() {
	if (!ok() || !impl_->meshReady) return -1;
	Impl &d = *impl_;

	// Snapshot pre-step positions for the backward pass. Both buffers are
	// host-visible and nothing is in flight here, so a host copy is
	// equivalent to the Metal backend's blit and avoids a second submit.
	if (d.positionsPreStep.valid()) {
		std::memcpy(d.positionsPreStep.mapped, d.positions.mapped,
				size_t(d.positions.size));
	}

	VkCommandBuffer cmd = d.begin();
	if (cmd == VK_NULL_HANDLE) return -1;

	for (uint32_t c = 0; c < d.numColors; ++c) {
		const uint32_t offset = d.colorStart[c];
		const uint32_t count = d.colorCount[c];
		if (count == 0) continue;

		const VbdInitParams ip{d.invHSqCached, offset, count};
		if (!d.dispatch(cmd, "vbd_init", count, &ip, sizeof(ip))) return -1;

		const VbdGatherParams gp{offset, count};

		// The force kernels run over ALL constraints on every color, not
		// just this color's: they must observe positions already updated
		// by earlier colors. Same as the Metal backend.
		if (d.nSprings) {
			if (!d.dispatch(cmd, "spring_force", d.nSprings)) return -1;
			if (!d.dispatch(cmd, "vbd_gather_spring", count, &gp, sizeof(gp))) return -1;
		}
		if (d.nAttach) {
			if (!d.dispatch(cmd, "attachment_force_al", d.nAttach)) return -1;
			if (!d.dispatch(cmd, "vbd_gather_attachment", count, &gp, sizeof(gp)))
				return -1;
		}
		if (d.nTri) {
			if (!d.dispatch(cmd, "triangle_membrane_force_al", d.nTri)) return -1;
			if (!d.dispatch(cmd, "vbd_gather_triangle", count, &gp, sizeof(gp))) return -1;
		}
		if (d.nBend) {
			if (!d.dispatch(cmd, "triangle_bending_force_al", d.nBend)) return -1;
			if (!d.dispatch(cmd, "vbd_gather_bending", count, &gp, sizeof(gp))) return -1;
		}

		const VbdSolveApplyParams sp{offset, count};
		if (!d.dispatch(cmd, "vbd_solve_apply", count, &sp, sizeof(sp))) return -1;
	}

	return d.endAndWait(cmd) ? 0 : -1;
}

// AVBD's Eq. 16 penalty ramp, on top of the dual ascent: beta scales how
// fast the penalty climbs with the constraint violation, clamped to
// penaltyMax.
//
// THE DEFAULT IS OFF (beta = 0), which reproduces the fixed-gamma
// behaviour these kernels had before, and that is a deliberate choice
// rather than caution for its own sake.
//
// Upstream's betaLin is 10000 (avbd-demo3d solver.cpp defaultParams),
// but upstream does not ACCUMULATE the dual. Its hard-constraint update
// is a replacement, lambda <- K*C + lambda evaluated fresh each
// iteration, and it decays lambda and the penalty between steps by
// alpha*gamma (Eq. 19). Ours accumulates lambda += gamma*C with no
// decay. Ramping gamma underneath an accumulating lambda is a different
// system, and it diverges at upstream's beta:
//
//   1 attachment + 1 triangle + 1 bending, 64 iterations
//     beta      0     1    10   100   1000   10000
//     result   ok    ok    ok    ok    NaN     NaN
//
//   and the boundary does not move with penaltyMax (1e4, 1e6, 1e10 all
//   give the same column), so it is the growth RATE against the
//   accumulating dual, not gamma reaching its ceiling.
//
// So the ramp is implemented, tested and off. Turning it on safely means
// adopting upstream's lambda form and Eq. 19 decay as well -- the whole
// scheme, not half of it -- which is the next rung, not this one.
// Defaulting to a value that merely happened to survive one 4-vertex
// fixture would be picking a number, not making a decision.
//
// AVBD_BETA enables it; test_avbd_penalty_ramp measures what it buys
// (510x better attachment satisfaction at 64 iterations) and pins the
// divergence boundary above so a future change has to confront it.
struct DualUpdateParams {
	float beta;
	float penaltyMax;
	uint32_t count;
};

// Deliberately NOT cached in a static. A cached value can only be set
// once per process, which would force the ramp-on and ramp-off arms of
// test_avbd_penalty_ramp into separate runs and make them incomparable.
// One getenv per dual dispatch is nothing beside the dispatch itself.
static float avbdBeta() {
	if (const char *e = std::getenv("AVBD_BETA")) return float(std::atof(e));
	return 0.0f;
}

static float avbdPenaltyMax() {
	if (const char *e = std::getenv("AVBD_PENALTY_MAX")) return float(std::atof(e));
	return 1.0e10f;
}

int AvbdSolver::stepDualAttachments() {
	if (!ok() || impl_->nAttach == 0) return 0;
	Impl &d = *impl_;
	VkCommandBuffer cmd = d.begin();
	if (cmd == VK_NULL_HANDLE) return -1;
	const DualUpdateParams p{avbdBeta(), avbdPenaltyMax(), d.nAttach};
	if (!d.dispatch(cmd, "attachment_dual_update", d.nAttach, &p, sizeof(p))) return -1;
	return d.endAndWait(cmd) ? 0 : -1;
}

int AvbdSolver::stepDualMembrane() {
	if (!ok() || impl_->nTri == 0) return 0;
	Impl &d = *impl_;
	const DualUpdateParams p{avbdBeta(), avbdPenaltyMax(), d.nTri};
	VkCommandBuffer cmd = d.begin();
	if (cmd == VK_NULL_HANDLE) return -1;
	if (!d.dispatch(cmd, "triangle_membrane_dual_update", d.nTri, &p, sizeof(p))) return -1;
	return d.endAndWait(cmd) ? 0 : -1;
}

int AvbdSolver::stepDualBending() {
	if (!ok() || impl_->nBend == 0) return 0;
	Impl &d = *impl_;
	const DualUpdateParams p{avbdBeta(), avbdPenaltyMax(), d.nBend};
	VkCommandBuffer cmd = d.begin();
	if (cmd == VK_NULL_HANDLE) return -1;
	if (!d.dispatch(cmd, "triangle_bending_dual_update", d.nBend, &p, sizeof(p))) return -1;
	return d.endAndWait(cmd) ? 0 : -1;
}

// ---------------------------------------------------------------------
// Self collision
// ---------------------------------------------------------------------

void AvbdSolver::uploadSelfCollisionRadii(const float *radiiIn, uint32_t maxNeighborsPerVert) {
	if (!ok() || !impl_->meshReady) return;
	Impl &d = *impl_;
	const uint32_t vcap = capacityFor(d.nVerts);
	d.selfCollK = maxNeighborsPerVert;
	if (!d.alloc(d.radii, "radii", VkDeviceSize(vcap) * sizeof(float))) return;
	if (!d.alloc(d.neighbors, "neighbors",
				VkDeviceSize(vcap) * maxNeighborsPerVert * sizeof(uint32_t))) {
		return;
	}
	std::memcpy(d.radii.mapped, radiiIn, size_t(d.nVerts) * sizeof(float));
	d.selfCollReady = true;
}

int AvbdSolver::detectSelfCollisions(std::vector<std::pair<uint32_t, uint32_t>> &out_pairs) {
	out_pairs.clear();
	if (!ok() || !impl_->meshReady || !impl_->selfCollReady) return -1;
	Impl &d = *impl_;

	VkCommandBuffer cmd = d.begin();
	if (cmd == VK_NULL_HANDLE) return -1;
	const SelfCollisionScanParams sp{d.nVerts, d.selfCollK};
	if (!d.dispatch(cmd, "self_collision_scan", d.nVerts, &sp, sizeof(sp))) return -1;
	if (!d.endAndWait(cmd)) return -1;

	const uint32_t *nb = d.neighbors.u32();
	for (uint32_t i = 0; i < d.nVerts; ++i) {
		for (uint32_t k = 0; k < d.selfCollK; ++k) {
			const uint32_t j = nb[i * d.selfCollK + k];
			if (j == kSentinel) break;
			if (j > i) out_pairs.emplace_back(i, j);  // dedupe mirrored pairs
		}
	}
	return 0;
}

// ---------------------------------------------------------------------
// Readback
// ---------------------------------------------------------------------

void AvbdSolver::readPositions(std::vector<float> &out) const {
	if (!ok()) {
		out.clear();
		return;
	}
	readVec3Padded(impl_->positions, out, impl_->nVerts);
}

void AvbdSolver::readScratch(std::vector<float> &g_out, std::vector<float> &h_out) const {
	if (!ok()) {
		g_out.clear();
		h_out.clear();
		return;
	}
	readVec3Padded(impl_->gScratch, g_out, impl_->nVerts);
	readScalars(impl_->hScratch, h_out, 6 * impl_->nVerts);
}

void AvbdSolver::readSpringForce(std::vector<float> &gradA_out,
		std::vector<float> &hess_out) const {
	if (!ok()) {
		gradA_out.clear();
		hess_out.clear();
		return;
	}
	readVec3Padded(impl_->springGradA, gradA_out, impl_->nSprings);
	readScalars(impl_->springHess, hess_out, 6 * impl_->nSprings);
}

// ---------------------------------------------------------------------
// Reverse-mode adjoint
// ---------------------------------------------------------------------

bool AvbdSolver::Impl::ensureBackwardBuffers() {
	if (backwardReady) return true;

	const uint32_t vcap = capacityFor(nVerts);
	const VkDeviceSize v3 = VkDeviceSize(vcap) * 4 * sizeof(float);
	const VkDeviceSize v1 = VkDeviceSize(vcap) * sizeof(float);
	const VkDeviceSize vh = VkDeviceSize(vcap) * 6 * sizeof(float);

	if (!alloc(vPositionsLoss, "v_out", v3)) return false;
	if (!alloc(vG, "v_g", v3)) return false;
	if (!alloc(vH, "v_H", vh)) return false;
	if (!alloc(deltaX, "deltaX", v3)) return false;
	if (!alloc(vPositionsGrad, "vPositionsGrad", v3)) return false;
	if (!alloc(vPositionsInit, "v_x", v3)) return false;
	if (!alloc(vPredicted, "v_y", v3)) return false;
	if (!alloc(vMass, "v_mass", v1)) return false;
	if (!alloc(hScratchJunk, "hScratchJunk", vh)) return false;

	if (nSprings) {
		const uint32_t c = capacityFor(nSprings);
		if (!alloc(vSpringGradA, "v_springGradA", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
		if (!alloc(vSpringHess, "v_springHess", VkDeviceSize(c) * 6 * sizeof(float)))
			return false;
		if (!alloc(vSpringPd, "vSpringPd", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
		if (!alloc(vSpringRestLen, "vSpringRestLen", VkDeviceSize(c) * sizeof(float)))
			return false;
		if (!alloc(vSpringStiff, "vSpringStiff", VkDeviceSize(c) * sizeof(float)))
			return false;
	}
	if (nAttach) {
		const uint32_t c = capacityFor(nAttach);
		if (!alloc(vAttachGradV, "v_attachGradV", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
		if (!alloc(vAttachHessScalar, "v_attachHessScalar", VkDeviceSize(c) * sizeof(float)))
			return false;
		if (!alloc(vAttachFixedPos, "vAttachFixedPos", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
		if (!alloc(vAttachStiff, "vAttachStiff", VkDeviceSize(c) * sizeof(float)))
			return false;
		if (!alloc(vAttachLambda, "vAttachLambda", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
	}
	if (nTri) {
		const uint32_t c = capacityFor(nTri);
		if (!alloc(vTriGrad, "v_triGrad", VkDeviceSize(c) * 3 * 4 * sizeof(float)))
			return false;
		if (!alloc(vTriHessScalar, "v_triHessScalar", VkDeviceSize(c) * 3 * sizeof(float)))
			return false;
		if (!alloc(vTriP, "vTriP", VkDeviceSize(c) * 3 * 4 * sizeof(float))) return false;
		if (!alloc(vTriStiff, "vTriStiff", VkDeviceSize(c) * sizeof(float))) return false;
		if (!alloc(vTriLambda0, "vTriLambda0", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
		if (!alloc(vTriLambda1, "vTriLambda1", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
	}
	if (nBend) {
		const uint32_t c = capacityFor(nBend);
		if (!alloc(vBendGrad, "v_bendGrad", VkDeviceSize(c) * 4 * 4 * sizeof(float)))
			return false;
		if (!alloc(vBendHessScalar, "v_bendHessScalar", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
		if (!alloc(vBendP, "vBendP", VkDeviceSize(c) * 4 * 4 * sizeof(float))) return false;
		if (!alloc(vBendNTarget, "vBendNTarget", VkDeviceSize(c) * sizeof(float)))
			return false;
		if (!alloc(vBendStiff, "vBendStiff", VkDeviceSize(c) * sizeof(float))) return false;
		if (!alloc(vBendLambda, "vBendLambda", VkDeviceSize(c) * 4 * sizeof(float)))
			return false;
	}

	backwardReady = true;
	return true;
}

int AvbdSolver::stepBackward(const float *v_positions_loss) {
	if (!ok() || !impl_->meshReady) return -1;
	Impl &d = *impl_;
	if (!d.ensureBackwardBuffers()) return -1;

	// Host prologue, mirroring the Metal backend: upload the incoming
	// cotangent, recover the forward step's position delta, and clear
	// the accumulator that the scatter phase adds into.
	uploadVec3Padded(d.vPositionsLoss, v_positions_loss, d.nVerts);
	{
		const float *post = d.positions.f32();
		const float *pre = d.positionsPreStep.f32();
		float *dx = d.deltaX.f32();
		for (uint32_t i = 0; i < d.nVerts; ++i) {
			dx[4 * i + 0] = post[4 * i + 0] - pre[4 * i + 0];
			dx[4 * i + 1] = post[4 * i + 1] - pre[4 * i + 1];
			dx[4 * i + 2] = post[4 * i + 2] - pre[4 * i + 2];
			dx[4 * i + 3] = 0.0f;
		}
	}
	std::memset(d.vPositionsGrad.mapped, 0, size_t(d.vPositionsGrad.size));
	std::memset(d.hScratchJunk.mapped, 0, size_t(d.hScratchJunk.size));

	// Every *_force_*_backward kernel must see the positions the forward
	// force kernels saw, i.e. pre-solve_apply. (With numColors > 1 this
	// is only exact for the first color -- a limitation inherited from
	// the Metal backend, see AvbdSolver.mm:1355.)
	const Override prePos{"positions", &d.positionsPreStep};

	VkCommandBuffer cmd = d.begin();
	if (cmd == VK_NULL_HANDLE) return -1;

	if (!d.dispatch(cmd, "vbd_solve_apply_backward", d.nVerts)) return -1;

	if (d.nSprings && !d.dispatch(cmd, "vbd_gather_spring_backward", d.nSprings)) return -1;
	if (d.nAttach && !d.dispatch(cmd, "vbd_gather_attachment_backward", d.nAttach))
		return -1;
	if (d.nTri && !d.dispatch(cmd, "vbd_gather_triangle_backward", 3 * d.nTri)) return -1;
	if (d.nBend && !d.dispatch(cmd, "vbd_gather_bending_backward", 4 * d.nBend)) return -1;

	// attachment_force_al_backward writes v_positions non-additively
	// (1:1 per attached vertex), so it has to land before the additive
	// scatters below or it would clobber them.
	if (d.nAttach &&
			!d.dispatch(cmd, "attachment_force_al_backward", d.nAttach, nullptr, 0,
					{prePos})) {
		return -1;
	}
	if (d.nSprings &&
			!d.dispatch(cmd, "spring_force_backward", d.nSprings, nullptr, 0, {prePos})) {
		return -1;
	}
	if (d.nTri &&
			!d.dispatch(cmd, "triangle_membrane_force_al_backward", d.nTri, nullptr, 0,
					{prePos})) {
		return -1;
	}
	if (d.nBend &&
			!d.dispatch(cmd, "triangle_bending_force_al_backward", d.nBend, nullptr, 0,
					{prePos})) {
		return -1;
	}

	const VbdInitBackwardParams ibp{d.invHSqCached};
	if (!d.dispatch(cmd, "vbd_init_backward", d.nVerts, &ibp, sizeof(ibp), {prePos})) {
		return -1;
	}

	// Scatter the per-constraint position cotangents into v_positions by
	// reusing the FORWARD gather kernels -- their role buffers already
	// carry the per-corner sign convention. colorOffset 0 and a full nV
	// dispatch: coloring is irrelevant for an additive scatter.
	const VbdGatherParams gz{0u, d.nVerts};
	if (d.nSprings &&
			!d.dispatch(cmd, "vbd_gather_spring", d.nVerts, &gz, sizeof(gz),
					{{"springGradA", &d.vSpringPd},
							{"springHess", &d.vSpringHess},
							{"gScratch", &d.vPositionsGrad},
							{"hScratch", &d.hScratchJunk}})) {
		return -1;
	}
	if (d.nTri &&
			!d.dispatch(cmd, "vbd_gather_triangle", d.nVerts, &gz, sizeof(gz),
					{{"triGrad", &d.vTriP},
							{"triHessScalar", &d.vTriHessScalar},
							{"gScratch", &d.vPositionsGrad},
							{"hScratch", &d.hScratchJunk}})) {
		return -1;
	}
	if (d.nBend &&
			!d.dispatch(cmd, "vbd_gather_bending", d.nVerts, &gz, sizeof(gz),
					{{"bendGrad", &d.vBendP},
							{"bendHessScalar", &d.vBendHessScalar},
							{"gScratch", &d.vPositionsGrad},
							{"hScratch", &d.hScratchJunk}})) {
		return -1;
	}

	if (!d.endAndWait(cmd)) return -1;

	// Host epilogue. Two contributions land on dL/dx here.
	//
	// 1. The inertial path, v_x = w * v_g from vbd_init_backward.
	//
	// 2. The DIRECT path. vbd_solve_apply computes
	//    positions[v] = p + dx, so x_out depends on x through the
	//    identity as well as through g, and dL/dx picks up the
	//    incoming cotangent v_out unchanged. This term was missing,
	//    which is why dL/dx was wrong even with no constraints at all:
	//    with inertia alone H = w*I and dx = predicted - x, so
	//    x_out = predicted exactly and dL/dx must be ZERO -- it comes
	//    out as v_out + w*v_g, the two cancelling. Dropping v_out left
	//    exactly -v_out behind.
	{
		float *grad = d.vPositionsGrad.f32();
		const float *init = d.vPositionsInit.f32();
		const float *vout = d.vPositionsLoss.f32();
		for (uint32_t i = 0; i < d.nVerts; ++i) {
			for (int c = 0; c < 3; ++c) {
				grad[4 * i + c] += init[4 * i + c] + vout[4 * i + c];
			}
		}
	}
	return 0;
}

void AvbdSolver::readPositionsGrad(std::vector<float> &out) const {
	if (!ok()) {
		out.clear();
		return;
	}
	readVec3Padded(impl_->vPositionsGrad, out, impl_->nVerts);
}

void AvbdSolver::readMassGrad(std::vector<float> &out) const {
	if (!ok()) {
		out.clear();
		return;
	}
	readScalars(impl_->vMass, out, impl_->nVerts);
}

void AvbdSolver::readPredictedGrad(std::vector<float> &out) const {
	if (!ok()) {
		out.clear();
		return;
	}
	readVec3Padded(impl_->vPredicted, out, impl_->nVerts);
}

void AvbdSolver::readSpringGrad(std::vector<float> &restLen_grad,
		std::vector<float> &stiff_grad) const {
	if (!ok()) {
		restLen_grad.clear();
		stiff_grad.clear();
		return;
	}
	readScalars(impl_->vSpringRestLen, restLen_grad, impl_->nSprings);
	readScalars(impl_->vSpringStiff, stiff_grad, impl_->nSprings);
}

void AvbdSolver::readAttachGrad(std::vector<float> &fixedPos_grad,
		std::vector<float> &stiff_grad, std::vector<float> &lambda_grad) const {
	if (!ok()) {
		fixedPos_grad.clear();
		stiff_grad.clear();
		lambda_grad.clear();
		return;
	}
	readVec3Padded(impl_->vAttachFixedPos, fixedPos_grad, impl_->nAttach);
	readScalars(impl_->vAttachStiff, stiff_grad, impl_->nAttach);
	readVec3Padded(impl_->vAttachLambda, lambda_grad, impl_->nAttach);
}

void AvbdSolver::readTriGrad(std::vector<float> &stiff_grad,
		std::vector<float> &lambda0_grad, std::vector<float> &lambda1_grad) const {
	if (!ok()) {
		stiff_grad.clear();
		lambda0_grad.clear();
		lambda1_grad.clear();
		return;
	}
	readScalars(impl_->vTriStiff, stiff_grad, impl_->nTri);
	readVec3Padded(impl_->vTriLambda0, lambda0_grad, impl_->nTri);
	readVec3Padded(impl_->vTriLambda1, lambda1_grad, impl_->nTri);
}

void AvbdSolver::readBendGrad(std::vector<float> &nTarget_grad,
		std::vector<float> &stiff_grad, std::vector<float> &lambda_grad) const {
	if (!ok()) {
		nTarget_grad.clear();
		stiff_grad.clear();
		lambda_grad.clear();
		return;
	}
	readScalars(impl_->vBendNTarget, nTarget_grad, impl_->nBend);
	readScalars(impl_->vBendStiff, stiff_grad, impl_->nBend);
	readVec3Padded(impl_->vBendLambda, lambda_grad, impl_->nBend);
}

}  // namespace cloth
