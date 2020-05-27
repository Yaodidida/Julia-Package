using Base.Iterators: partition
using DelimitedFiles
using Flux
using Flux.Data.MNIST
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, glorot_normal
using Images
using Statistics
using Printf

directory = "output"
result = "results"

mutable struct DCGAN1
    batch_size::Int64
    noise_dim::Int64
    channels::Int64
    
    generator::Chain
    discriminator::Chain

    generator_optimizer
    discriminator_optimizer

    data::Vector{<: AbstractArray{Float32, 4}}

    output_size::Pair{Int64, Int64}
    output_noise::AbstractMatrix{Float32}

    train_steps::Int64
    verbose_freq::Int64
    
    gen_losshist::Vector{Float32}
    dis_loss_hist::Vector{Float32}
end

function DCGAN1(; image_vector::Vector{<: AbstractMatrix}, noise_dim::Int64, channels::Int64, batch_size::Int64, 
    generator::Chain, discriminator::Chain,
    output_size::Pair{Int64, Int64}, verbose_freq::Int64)

    @assert (channels == 1) || (channels == 3)

    data = [reshape(reduce(hcat, channelview.(xs)), 28, 28, 1, :) for xs in partition(image_vector, batch_size)]
    data = [2f0 .* gpu(Float32.(xs)) .- 1f0 for xs in data]

    output_noise = randn(Float32, noise_dim, prod(output_size))  

    DCGAN1(noise_dim, channels, batch_size, generator, discriminator, ADAM(0.0001f0), ADAM(0.0001f0), data, 
        output_size, output_noise, 0, verbose_freq, Vector{Float32}(), Vector{Float32}())
end


function Generator()
    noise_dim = 100
    channels = 1
    return Chain(
        Dense(noise_dim, 7 * 7 * 256; initW = glorot_normal),
        BatchNorm(7 * 7 * 256, relu),
        x->reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; init = glorot_normal, stride = 1, pad = 2),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; init = glorot_normal, stride = 2, pad = 1),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1, tanh; init = glorot_normal, stride = 2, pad = 1),
        )
end

function Discriminator()
    channels = 1
    return Chain(
        Conv((4, 4), channels => 64, leakyrelu; init = glorot_normal, stride = 2, pad = 1),
        Dropout(0.25),
        Conv((4, 4), 64 => 128, leakyrelu; init = glorot_normal, stride = 2, pad = 1),
        Dropout(0.25), 
        x->reshape(x, 7 * 7 * 128, :),
        Dense(7 * 7 * 128, 1; initW = glorot_normal))
end

function generator_loss(fake_output)
    loss = mean(logitbinarycrossentropy.(fake_output, 1f0))
    return loss 
end

function discriminator_loss(real_output, fake_output)
    real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
    fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
    loss = 0.5f0 * (real_loss +  fake_loss)
    return loss
end

function convert_to_image(image_array::Matrix{Float32}, channels::Int64)
    image_array = @. (image_array + 1f0) / 2f0
    if channels == 1
        return Gray.(image_array)
    else
        return colorview(RGB, image_array)
    end
end

function save_image(dcgan::DCGAN1)
    @eval Flux.istraining() = false
    fake_images = dcgan.generator(dcgan.output_noise)
    @eval Flux.istraining() = true
    h, w, _, _ = size(fake_images)
    rows, cols = dcgan.output_size.first, dcgan.output_size.second
    tile_image = Matrix{Float32}(undef, h * rows, w * cols)
    for n in 0:prod(dcgan.output_size) - 1
        j = n ÷ rows
        i = n % cols
        tile_image[j * h + 1:(j + 1) * h, i * w + 1:(i + 1) * w] = fake_images[:, :, :, n + 1] |> cpu
    end
    image = convert_to_image(tile_image, dcgan.channels)
    save(@sprintf("%s/steps_%06d.png", result, dcgan.train_steps), image)
end

function train_discriminator!(dcgan::DCGAN1, batch::AbstractArray{Float32, 4})
    noise = randn(Float32, dcgan.noise_dim, dcgan.batch_size)  
    fake_input = dcgan.generator(noise)
    loss(m) = discriminator_loss(m(batch), m(fake_input))
    disc_grad = gradient(()->loss(dcgan.discriminator), Flux.params(dcgan.discriminator))
    update!(dcgan.discriminator_optimizer, Flux.params(dcgan.discriminator), disc_grad)
    return loss(dcgan.discriminator)
end

function train_generator!(dcgan::DCGAN1, batch::AbstractArray{Float32, 4})
    noise = randn(Float32, dcgan.noise_dim, dcgan.batch_size)  
    loss(m) = generator_loss(dcgan.discriminator(m(noise)))
    gen_grad = gradient(()->loss(dcgan.generator), Flux.params(dcgan.generator))
    update!(dcgan.generator_optimizer, Flux.params(dcgan.generator), gen_grad)
    return loss(dcgan.generator)
end

function train!(dcgan::DCGAN1, epochs::Integer)
    for ep in 1:epochs
        @info "epoch $ep"
        for batch in dcgan.data
            disc_loss = train_discriminator!(dcgan, batch)
            gen_loss = train_generator!(dcgan, batch)

            if dcgan.train_steps % dcgan.verbose_freq == 0
                push!(dcgan.dis_loss_hist, disc_loss)
                push!(dcgan.gen_losshist, gen_loss)
                @info("Train step $(dcgan.train_steps), Discriminator loss: $(disc_loss), Generator loss: $(gen_loss)")
                # create fake images for output
                save_image(dcgan)
            end
            dcgan.train_steps += 1
        end
    end
end


function main()
    if !isdir(directory)
        mkdir(directory)
    end

    if !isdir(result)
        mkdir(result)
    end

    noise_dim = 100
    channels = 1 

    dcgan = DCGAN1(; image_vector = MNIST.images(), noise_dim = noise_dim, 
        channels = channels, batch_size = 128,
        generator = generator, discriminator = discriminator,
        output_size = 6=>6, verbose_freq = 100)
    train!(dcgan, 30)

    open(@sprintf("%s/discriminator_loss.txt", result), "w") do io
        writedlm(io, dcgan.dis_loss_hist)
    end

    open(@sprintf("%s/generator_loss.txt", result), "w") do io
        writedlm(io, dcgan.gen_losshist)
    end

end

function img_load(img_name)
    img = load(img_name)
    img = imresize(img, 64, 64)
    img = reshape(Float64.(channelview(img)), 64, 64, 3)
    return img
end

function load_dataset_as_batches(path, BATCH_SIZE)
    data = []
    for r in readdir(path)
        img_path = string(path, r)
        push!(data, img_load(img_path))
    end
    num_images = length(data)
    #println(num_images)
    batched_data = []
    for x in partition(data, BATCH_SIZE)
        x = reshape(cat(x..., dims = 4), 64, 64, 3, BATCH_SIZE)
        push!(batched_data, x)
    end
    return batched_data
end

main()
