def circuit(self, inputs, weights):

        for i in range(self.depth):
            # Apply rotation gates using weights
            for j in range(self.n_qubits):
                qml.RY(weights[i, j, 0], wires=j)
                qml.RZ(weights[i, j, 1], wires=j)
                qml.RY(weights[i, j, 2], wires=j)
                qml.RZ(weights[i, j, 3], wires=j)
            
            # Apply controlled rotations between neighboring qubits
            for j in range(self.n_qubits - 1):  # Controlled rotations between adjacent qubits
                qml.CRY(weights[i, j, 4], wires=[j, j+1])
                qml.CRZ(weights[i, j, 5], wires=[j, j+1])
        
            # Apply controlled rotations between the last and the first qubit
            qml.CRY(weights[i, self.n_qubits - 1, 4], wires=[self.n_qubits - 1, 0])
            qml.CRZ(weights[i, self.n_qubits - 1, 5], wires=[self.n_qubits - 1, 0])
            qml.Barrier()
        return qml.probs(wires=list(range(self.n_qubits)))





# Train the encoder
            self.toggle_optimizer(optE)
            self.toggle_optimizer(optG)
            optE.zero_grad()
            optG.zero_grad()

            # Recompute recon_imgs for the encoder
            mu, log_var, z, recon_imgs = self.model.vae_forward(real_data)

            recon_loss = F.mse_loss(recon_imgs, real_data,reduction='sum') / batch_size
            prior_loss = self.normal_kld(mu, log_var)
            errE =  prior_loss + recon_loss
            
            fake_validity = critic(recon_imgs)
            errG = - torch.mean(fake_validity)

            loss = errE + errG

            self.manual_backward(loss)

            optE.step()
            optG.step()


            self.untoggle_optimizer(optE)
            self.untoggle_optimizer(optG) 

            
            self.log('prior_loss', prior_loss)
            self.log('e_loss', errE, prog_bar=True)
            
            self.log("train_g_loss_step", errG, prog_bar=True)
  



  # Train the encoder
            self.toggle_optimizer(optE)
            optE.zero_grad()
            
            # Recompute recon_imgs for the encoder
            mu, log_var, z, recon_imgs = self.model.vae_forward(real_data)
            
            recon_loss = F.mse_loss(recon_imgs, real_data,reduction='sum') / batch_size
            prior_loss = self.normal_kld(mu, log_var) / batch_size

            errE =  prior_loss + recon_loss
            self.manual_backward(errE)
            optE.step()

            self.log('prior_loss', prior_loss)
            self.log('e_loss', errE, prog_bar=True)

            self.untoggle_optimizer(optE)

            self.toggle_optimizer(optG)

            mu, log_var, z, fake_data = self.model.vae_forward(real_data)
            # Loss measures generator's ability to fool the discriminator,Train on fake images
            recon_loss = F.mse_loss(fake_data, real_data, reduction='sum') / batch_size

            fake_validity = critic(fake_data)
            errG = - torch.mean(fake_validity) + self.recon_weight * recon_loss
        
            self.manual_backward(errG)
            optG.step()
            self.log("train_g_loss_step", errG, prog_bar=True)
  
            self.untoggle_optimizer(optG) 