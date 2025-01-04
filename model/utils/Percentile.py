import torch
import warnings
warnings.filterwarnings("ignore")
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()
    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)
    def forward(self, input, percentiles):
        input = torch.flatten(input) # 输入数据扁平化处理（压缩为一维向量）
        input_dtype = input.dtype #
        input_shape = input.shape
        # 判断分位数是否为整数，若为整数，则将数据类型转化为元组
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        # 转化数据的数据类型
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        # 与上诉类似
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        # 类似
        input = input.double()
        # 转化数据运用的计算容器
        percentiles = percentiles.to(input.device).double()
        # 调整数据维度，行为input.shape[0]，列为自动计算的剩余维度
        input = input.view(input.shape[0], -1)
        # 将排序结果和排序索引分别储存下来
        in_sorted, in_argsort = torch.sort(input, dim=0)
        # 计算百分位数的值
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions) # 计算百分位数的下限
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1 # 上限索引
        weight_ceiled = positions-floored # 计算每个百分位数的权重
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None] # 计算其对应的值
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        # 用于反向传播的值和索引以及权重保存到类的属性
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        # 将计算的结果重新调整形状，使其与输入张量的形状相同，并将其转换回输入的原始数据类型
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)
    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        # 获取之前保存的信息
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector
        # 列偏移量，为一维向量，其每个元素都是对应输入张量列的索引
        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        # 对排序后的输入、下限索引和上限索引分别与列偏移量相乘并展平之后，它们可以被用作索引，用来从梯度输出和上下限权重中取出对应元素
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        # 创建一个全零的张量grad_input，大小与in_argsort的大小相同
        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        # 然后根据floored和ceiled索引从grad_output和对应的权重中取出元素，加到grad_input中
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)
        # 最后，将grad_input重新调整为输入的形状，然后返回
        grad_input = grad_input.view(*input_shape)
        return grad_input