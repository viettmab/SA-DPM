import torch

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    
def calculate_alpha(beta):
    alphas_cumprod = (1 - beta).cumprod(dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1).to(alphas_cumprod.device), alphas_cumprod[:-1]], dim=0)
    alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.zeros(1).to(alphas_cumprod.device)], dim=0)
    recip_noise_coef = torch.sqrt(1-alphas_cumprod) * torch.sqrt(1-beta) / beta
    return {
        "alphas_cumprod": alphas_cumprod, 
        "alphas_cumprod_prev": alphas_cumprod_prev, 
        "alphas_cumprod_next": alphas_cumprod_next, 
        "recip_noise_coef": recip_noise_coef,
    }

def extract_into_tensor(x, t):
    return x.index_select(0, t.long()).view(-1, 1, 1, 1)

def sequence_aware_loss(model,
                    x0: torch.Tensor,
                    t: torch.LongTensor,
                    e: torch.Tensor,
                    b: torch.Tensor, num_consecutive_steps=2, lamda=0.1, keepdim=False):
    coef = calculate_alpha(b)
    at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
    xt = x0 * torch.sqrt(at_bar) + e * torch.sqrt(1.0 - at_bar)
    eps_prediction = model(xt, t.float())
    mse = (e - eps_prediction).square().sum(dim=(1, 2, 3)).mean(dim=0)
    sa_loss = e - eps_prediction
    t_k = t
    for k in range(1,num_consecutive_steps):
        at_bar_prev_k = extract_into_tensor(coef["alphas_cumprod_prev"],t_k)
        e_k = torch.randn_like(e)
        xt_k = x0 * torch.sqrt(at_bar_prev_k) + e_k * torch.sqrt(1.0 - at_bar_prev_k)
        mask_k = (t >= k).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_k = torch.clamp(t - k, min=0)
        eps_prediction_k = model(xt_k, t_k.float())
        sa_loss += (e_k - eps_prediction_k)*mask_k
    sa_loss = 1/num_consecutive_steps * sa_loss.square().sum(dim=(1, 2, 3)).mean(dim=0)
    total_loss = mse + lamda * sa_loss
    return total_loss

def get_loss_value(model,
                    x0: torch.Tensor,
                    t: torch.LongTensor,
                    e: torch.Tensor,
                    b: torch.Tensor):
    coef = calculate_alpha(b)
    at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
    xt = x0 * torch.sqrt(at_bar) + e * torch.sqrt(1.0 - at_bar)
    eps_prediction = model(xt, t.float())
    dic = {}
    step = int(t.mean().item())-1
    dic[step+1] = torch.zeros_like(eps_prediction)
    for i in range(step,0,-1): # LOOP: step-1 timesteps
        print(i)
        at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
        at_bar_prev = extract_into_tensor(coef["alphas_cumprod_prev"],t)
        
        x0_pred = (xt - eps_prediction * torch.sqrt(1 - at_bar)) / torch.sqrt(at_bar)
        t = torch.clamp(t - 1.0, min=0)
        
        eta = 1
        sigma = eta * torch.sqrt((1 - at_bar / at_bar_prev) * (1 - at_bar_prev) / (1 - at_bar))
        c2 = torch.sqrt(1 - at_bar_prev - sigma ** 2)

        xt = torch.sqrt(at_bar_prev) * x0_pred + c2 * eps_prediction + sigma * torch.randn_like(xt)
        eps_prediction = model(xt, t.float())
        eps_target = (xt-torch.sqrt(at_bar_prev)*x0) / torch.sqrt(1 - at_bar_prev)
        mse = (eps_prediction-eps_target).mean(dim=0)
        at_1 = extract_into_tensor(1-b,t)
        d_t_1 = mse * (1-at_1)/torch.sqrt(at_1*(1-at_bar_prev))

        dic[i] = d_t_1 + dic[i+1]*torch.sqrt(at_1)*(1-extract_into_tensor(coef["alphas_cumprod_prev"],torch.clamp(t - 1.0, min=0)))/(1-at_bar_prev)
    for key in dic.keys():
        dic[key] = torch.mean(dic[key],0)
    return dic

loss_registry = {
    'simple': noise_estimation_loss,
    'sa': sequence_aware_loss,
    "get_loss_value": get_loss_value,
}
