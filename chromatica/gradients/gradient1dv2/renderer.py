"""
I'll outline the steps to build the renderer for the 1D gradient module.
Let 0 <= u <= 1 be the interpolation coefficient.
For a simple linear gradient between two colors C0 and C1, the only thing we need to do is apply
the spatial transforms, the per-channel transforms, and then do a linear interpolation.

So grosso modo Render(u) = Stack of [
Transform_channel_i(
       Global_Transform(u))
    for i in range(num_channels)
       ]

Now for a multi-segment gradient, we need to first figure out which segment u belongs to.
 u -> Segment a.k.a. S_k, u_local
 Render(u) = Stack of [
    Transform_channel_ik(
        Global_Transform_k(
        Locate(Global_Transform(u))
        ))
    for i in range(num_channels)
]
u -> global transform -> locate segment k and local u -> segment k global transform -> per-channel transforms
Selector -> graps set(u) -> segment k, u_local -> segment k global transform -> per-channel transforms
"""