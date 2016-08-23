#ifndef HERCULES_RIGID_BODY_H
#define HERCULES_RIGID_BODY_H

#include <ponos.h>

namespace hercules {

	/* Auxiliary struct for <RigidBody>.
	 * Stores the derivatives of the quantities of the body
	 * needed by the solver.
	 */
	struct DdtBodyState {
		// space derivative
		ponos::vec3 dxt;
		// linear momentum derivative
		ponos::vec3 dP;
		// angular momentum derivative
		ponos::vec3 dL;
		// rotation derivative
		ponos::quat dq;
	};
	/* Represents the state of a rigid body.
	 * Stores the state of a rigid body for physical simulations.
	 */
	class RigidBody {
  	public:
	 		RigidBody() {}
			virtual ~RigidBody() {}

			// constant quantities
			float density;
			float mass;
			ponos::mat3 IbodyInv;
			// state variables
			ponos::Point3 x;
			ponos::quat q;
			// linear momentum
			ponos::vec3 P;
			// angular momentum
			ponos::vec3 L;
			// derived quantities
			ponos::mat3 inertiaTensor;
			ponos::mat3 inertiaTensorInv;
			// rotation matrix
			ponos::mat3 R;
			// velocity
			ponos::vec3 v;
			// rotation
			ponos::vec3 omega;
			// computed quantities
			ponos::vec3 force;
			ponos::vec3 torque;

			/* Update state derived quantities.
			 * After the quantities of force and torque were updated, other
			 * quantities must be computed. This function assumes that <ComputeForceAndTorque> has already been called for the current step of the simulation.
			 */
			void update() {
				// compute new velocity v(t) = P(t) / M
				v = P / mass;
				// compute new rotation matrix
				R = ponos::normalize(q).toRotationMatrix();
				// I^-1(t) = R(t) Ibody^-1 R(t)^T
				inertiaTensorInv = R * IbodyInv * ponos::transpose(R);
				// w(t) = I^-1 L(t)
				omega = inertiaTensorInv * L;
			}

			void dxdt(double t, DdtBodyState &xdot) {
				// ddt state
				xdot.dxt = v;
				xdot.dq = .5f * (omega * q);
				// dP(t) = F(t)
				xdot.dP = force;
				// dL = torque(t)
				xdot.dL = torque;
			}

			void computeForceAndTorque() {}
	};

} // hercules namespace

#endif // HERCULES_RIGID_BODY_H

