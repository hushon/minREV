import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA

from typing import Callable, Tuple, Any



class coupling_block(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: Callable, G: Callable) -> Tuple[torch.Tensor, Any]:
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        """

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        """
        forward pass equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP
        """

        Y_1 = X_1 + F(X_2)

        # free memory since X_1 is now not needed
        del X_1

        Y_2 = X_2 + G(Y_1)

        # free memory since X_2 is now not needed
        del X_2
        
        ctx.F = F
        ctx.G = G
        ctx.prev_ctx = x.ctx if hasattr(x, "ctx") else None # 이전 레이어한테 input 복원한거 보내줘야하므로 필요

        output = torch.cat([Y_1, Y_2], dim=-1)
        output.ctx = ctx

        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        y = ctx.output  ## next block backward 때 또는 마지막 레이어 끝나고 사람이 넣어줌
        F, G = ctx.F, ctx.G

        # obtaining gradients dX_1 and dX_2 from the concatenated input
        Y_1, Y_2 = torch.chunk(y, 2, dim=-1)
        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1 = G(Y_1)
            g_Y_1.backward(dY_2)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dX_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2 = F(X_2)
            f_X_2.backward(dX_1)
        
        with torch.no_grad():
            dX_2 = dY_2 + X_2.grad

        dx = torch.cat([dX_1, dX_2], dim=-1)

        ## 여기서 다음 backward pass 에 X_1, X_2 를 전달해줘야함
        if ctx.prev_ctx is not None:
            X_1 = Y_1 - f_X_2
            x = torch.cat([X_1, X_2], dim=-1)
            ctx.prev_ctx.output = x.detach()

        return dx, None, None, None


class InvertibleBlock(nn.Module):
    """
    F 랑 G 는 임의의 모듈
    F랑 G를 coupling 구조에 끼워넣음. 
    backward pass 할때는 뒷쪽 블락에서 보내준 activation 값을 이용해 중간값 재계산
    Y_1 = X_1 + F(X_2)
    Y_2 = X_2 + G(Y_1)
    """
    def __init__(self, F: nn.Module, G: nn.Module, compute_inverse=True):
        ## compute_inverse 에 따라서 invertible 또는 vanilla 로 동작하도록 구혆필요
        super().__init__()
        self.F = F
        self.G = G
        self.compute_inverse = compute_inverse

    def forward(self, x):
        if self.compute_inverse:
            output = coupling_block.apply(x, self.F, self.G)
            ## prev_ctx 에 output 넣어주는걸 여기로 옮겨도 괜찮을듯
            ## ctx 가 autograd 인터페이스 밖으로 나오는게 좋지않음.. 메모리 free 안될수도 있음.
            ## 잘하면 output 에 backward hook 을 이용해서 ctx 를 숨길수도 있을듯 한데..
            ## 아님 외부에 context manager 를 둬서 거기다가 ctx 를 숨기는 방법도 있을듯
            ## with invertible_forward(enabled=True): 이런식으로
            ## 근데 그럼 다음 레이어가 invertible 한지를 알아야함 (input 을 되돌려줄수있는지)
            return output
        else:
            X_1, X_2 = torch.chunk(x, 2, dim=-1)
            Y_1 = X_1 + self.F(X_2)
            Y_2 = X_2 + self.G(Y_1)
            output = torch.cat([Y_1, Y_2], dim=-1), None
            return output, None


class InvertibleTransformerBlock(InvertibleBlock):
    def __init__(self, dim, num_heads):
        super().__init__(
            AttentionSubBlock(dim=dim, num_heads=num_heads),
            MLPSubblock(dim=dim)
        )


class InvertibleVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        patch_size=(2, 2),  # this patch size is used for CIFAR-10 --> (32 // 2)**2 = 256 sequence length
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size

        num_patches = (image_size[0] // self.patch_size[0]) * \
            (image_size[1] // self.patch_size[1])

        self.layers = nn.ModuleList(
            [
                InvertibleTransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.n_head
                )
                for _ in range(self.depth)
            ]
        )

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim)
        )

        # The two streams are concatenated and passed through a linear
        # layer for final projection. This is the only part of RevViT
        # that uses different parameters/FLOPs than a standard ViT model.
        # Note that fusion can be done in several ways, including
        # more expressive methods like in paper, but we use
        # linear layer + LN for simplicity.
        self.head = nn.Linear(2 * self.embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(2 * self.embed_dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embeddings

        # the two streams X_1 and X_2 are initialized identically with x and
        # concatenated along the last dimension to pass into the reversible blocks
        x = torch.cat([x, x], dim=-1)

        for layer in self.layers:
            x = layer(x)
        x.ctx.output = x.detach()

        # aggregate across sequence length
        x = x.mean(1)

        # head pre-norm
        x = self.norm(x)

        # pre-softmax logits
        x = self.head(x)

        # return pre-softmax logits
        return x


class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        num_heads
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # using vanilla attention for simplicity. To support adanced attention
        # module see pyslowfast.
        # Note that the complexity of the attention module is not a concern
        # since it is used blackbox as F block in the reversible logic and
        # can be arbitrary.
        self.attn = MHA(dim, num_heads, batch_first=True)

    def forward(self, x):
        # See MLP fwd pass for explanation.
        x = self.norm(x)
        out, _ = self.attn(x, x, x)
        return out


def main():
    """
    This is a simple test to check if the recomputation is correct
    by computing gradients of the first learnable parameters twice --
    once with the custom backward and once with the vanilla backward.

    The difference should be ~zero.
    """

    # insitantiating and fixing the model.
    model = RevViT()

    # random input, instaintiate and fixing.
    # no need for GPU for unit test, runs fine on CPU.
    x = torch.rand((1, 3, 32, 32))
    model = model

    # output of the model under reversible backward logic
    output = model(x)
    # loss is just the norm of the output
    loss = output.norm(dim=1)

    # computatin gradients with reversible backward logic
    # using retain_graph=True to keep the computation graph.
    loss.backward(retain_graph=True)

    # gradient of the patchification layer under custom bwd logic
    rev_grad = model.patch_embed.weight.grad.clone()

    # resetting the computation graph
    for param in model.parameters():
        param.grad = None

    # switching model mode to use vanilla backward logic
    model.no_custom_backward = True

    # computing forward with the same input and model.
    output = model(x)
    # same loss
    loss = output.norm(dim=1)

    # backward but with vanilla logic, does not need retain_graph=True
    loss.backward()

    # looking at the gradient of the patchification layer again
    vanilla_grad = model.patch_embed.weight.grad.clone()

    # difference between the two gradients is small enough.
    assert (rev_grad - vanilla_grad).abs().max() < 1e-6


def test():
    input = torch.rand(1, 16, requires_grad=True) ## TODO: 왜 requires_grad=True 가 필요한지 모르겠다
    F1 = torch.nn.Linear(8, 8)
    G1 = torch.nn.Linear(8, 8)
    F2 = torch.nn.Linear(8, 8)
    G2 = torch.nn.Linear(8, 8)

    F1.zero_grad()
    G1.zero_grad()
    F2.zero_grad()
    G2.zero_grad()
    block1 = InvertibleBlock(F1, G1)
    block2 = InvertibleBlock(F2, G2)
    output1, ctx = block1(input, None)
    output2, ctx = block2(output1, ctx)
    ctx.output = output2.detach()
    loss = output2.norm()
    loss.backward()
    breakpoint()

    F1.zero_grad()
    G1.zero_grad()
    F2.zero_grad()
    G2.zero_grad()
    block1 = CouplingBlockRef(F1, G1)
    block2 = CouplingBlockRef(F2, G2)
    output1 = block1(input)
    output2 = block2(output1)
    loss = output2.norm()
    loss.backward()
    breakpoint()



def test2():
    input = torch.rand(1, 3, 32, 32, requires_grad=True) ## TODO: 왜 requires_grad=True 가 필요한지 모르겠다
    model = InvertibleVisionTransformer()
    model.zero_grad()
    output = model(input)
    loss = output.norm()
    loss.backward()
    breakpoint()

    state_dict = model.state_dict()


    # input.grad.zero_()
    # model = RevViT()
    # model.load_state_dict(state_dict)
    # model.zero_grad()
    # output = model(input)
    # loss = output.norm()
    # # computatin gradients with reversible backward logic
    # # using retain_graph=True to keep the computation graph.
    # loss.backward(retain_graph=True)
    # breakpoint()


if __name__ == "__main__":
    # main()
    # test()
    test2()
